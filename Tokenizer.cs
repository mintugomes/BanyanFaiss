using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BanyanFaiss
{
    internal class MiniLMTokenizer
    {
        private TokenizerConfig _config;
        private int _maxLength;

        public MiniLMTokenizer(TokenizerConfig config, int maxLength = 128)
        {
            _config = config;
            _maxLength = maxLength;
        }

        public (long[] inputIds, long[] attentionMask) Tokenize(string text)
        {
            var tokens = new List<string> { "[CLS]" };

            foreach (var word in text.ToLower().Split(' '))
            {
                tokens.AddRange(TokenizeWordPiece(word));
            }

            tokens.Add("[SEP]");

            var inputIds = tokens.Select(t => (long)_config.GetId(t)).ToList();

            // Pad or truncate
            if (inputIds.Count > _maxLength)
                inputIds = inputIds.Take(_maxLength).ToList();
            else
                inputIds.AddRange(Enumerable.Repeat((long)_config.GetId("[PAD]"), _maxLength - inputIds.Count));

            var attentionMask = inputIds.Select(id => id == _config.GetId("[PAD]") ? 0L : 1L).ToArray();

            return (inputIds.ToArray(), attentionMask);
        }

        private List<string> TokenizeWordPiece(string word)
        {
            var subTokens = new List<string>();
            int start = 0;

            while (start < word.Length)
            {
                int end = word.Length;
                string subToken = null;

                while (start < end)
                {
                    var piece = (start > 0 ? "##" : "") + word.Substring(start, end - start);
                    if (_config.Contains(piece))
                    {
                        subToken = piece;
                        break;
                    }
                    end--;
                }

                if (subToken == null)
                {
                    subTokens.Add("[UNK]");
                    break;
                }

                subTokens.Add(subToken);
                start = end;
            }

            return subTokens;
        }
    }
}
