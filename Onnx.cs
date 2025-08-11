//using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace BanyanFaiss
{
    public static class Onnx
    {
        public enum EmbeddingMethods
        {
            CLS_Token,
            Mean_Pooling
        }
        public static float[] GetEmbedding(string text,EmbeddingMethods method)
        {
            float[] Embedding = null;

            var config = new TokenizerConfig(Config.Tokenizer);
            var tokenizer = new MiniLMTokenizer(config);

            //string inputText = "This is a test sentence.";
            var (inputIds, attentionMask) = tokenizer.Tokenize(text);

            // Load the ONNX model
            var session = new InferenceSession(Config.OnnxMiniLM);

            var inputTensor = new DenseTensor<long>(inputIds, new[] { 1, inputIds.Length });
            var maskTensor = new DenseTensor<long>(attentionMask, new[] { 1, attentionMask.Length });


            var tokenTypeIds = new DenseTensor<long>(new[] { 1, inputIds.Length });
            for (int i = 0; i < inputIds.Length; i++)
            {
                tokenTypeIds[0, i] = 0; // All zeros for single sentence
            }


            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", maskTensor),
                NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIds)
            };

            
            using var results = session.Run(inputs);
            //Embedding = results.First().AsEnumerable<float>().ToArray();
            //return Embedding;
            if (method == EmbeddingMethods.CLS_Token)
            {
                var output = results.First().AsTensor<float>();
                int hiddenSize = output.Dimensions[2];

                float[] embedding = new float[hiddenSize];
                for (int i = 0; i < hiddenSize; i++)
                {
                    embedding[i] = output[0, 0, i];
                }
                return embedding;
            }
            else if (method == EmbeddingMethods.Mean_Pooling)
            {
                var output = results.First().AsTensor<float>(); // [1, seq_len, hidden_size]
                int seqLen = output.Dimensions[1];
                int hiddenSize = output.Dimensions[2];

                float[] embedding = new float[hiddenSize];
                int validTokenCount = 0;

                for (int i = 0; i < seqLen; i++)
                {
                    if (attentionMask[i] == 1) // Only use non-padding tokens
                    {
                        validTokenCount++;
                        for (int j = 0; j < hiddenSize; j++)
                        {
                            embedding[j] += output[0, i, j];
                        }
                    }
                }

                // Divide by valid token count to get mean
                if (validTokenCount > 0)
                {
                    for (int j = 0; j < hiddenSize; j++)
                    {
                        embedding[j] /= validTokenCount;
                    }
                }

                return embedding;
            }
            else
                return new float[] { };
        }
    }
}
