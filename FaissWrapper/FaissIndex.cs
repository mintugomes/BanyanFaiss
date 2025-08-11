using BanyanFaiss;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace BanyanFaiss.CSharpWrapper
{
    public enum FaissIndexDescription
    {
        /// <summary>
        /// Brute-force exhaustive search using L2 (Euclidean) distance.
        /// </summary>
        Flat,

        /// <summary>
        /// Brute-force exhaustive search using Inner Product distance.
        /// </summary>
        FlatIP,

        /// <summary>
        /// Locality-Sensitive Hashing.
        /// </summary>
        LSH,

        /// <summary>
        /// Scalar Quantizer (e.g., SQ4, SQ6, SQ8 for 4, 6, or 8 bits per component).
        /// </summary>
        SQ,

        /// <summary>
        /// Product Quantizer (e.g., PQx where x is the number of bytes per vector).
        /// </summary>
        PQ,

        /// <summary>
        /// Inverted File Index with exact storage (e.g., IVF[nlist],Flat).
        /// </summary>
        IVFFlat,

        /// <summary>
        /// Inverted File Index with Product Quantization (e.g., IVF[nlist],PQ[bytes]).
        /// </summary>
        IVFPQ,

        /// <summary>
        /// Inverted File Index with Scalar Quantization (e.g., IVF[nlist],SQ[bits]).
        /// </summary>
        IVFSQ,

        /// <summary>
        /// Hierarchical Navigable Small World graph (e.g., HNSW[M],Flat).
        /// </summary>
        HNSWFlat,

        /// <summary>
        /// Principal Component Analysis for dimensionality reduction (e.g., PCA[dim]).
        /// This is typically a prefix for another index type.
        /// </summary>
        PCA,

        /// <summary>
        /// Optimized Product Quantizer (e.g., OPQ[subvecs]_[dim]).
        /// This is typically a prefix for another index type.
        /// </summary>
        OPQ,

        /// <summary>
        /// Re-ranking with Flat vectors (e.g., as a suffix like ,RFlat).
        /// </summary>
        RFlat,

        /// <summary>
        /// A generic placeholder for custom or highly composite index factories
        /// that combine multiple components not explicitly listed.
        /// </summary>
        CustomComposite
    }


    public class FaissIndex : IDisposable
    {
        private IntPtr _indexPtr;
        private bool _disposed = false;
        private int _dimensions = 384; //768;  // Match your embedding model output
        private FaissIndexDescription _description = FaissIndexDescription.Flat;

        public int Dimensions { get { return _dimensions; } }
        public FaissIndexDescription Description { get { return _description; } }  // Store the index description
        // Native method declarations
        private static class NativeMethods
        {
            [DllImport("FaissWrapper/Libs/win-x64/FaissWrapperNative.dll", CallingConvention = CallingConvention.Cdecl)]
            public static extern int faiss_index_factory(out IntPtr index, int d, string description);

            [DllImport("FaissWrapper/Libs/win-x64/FaissWrapperNative.dll", CallingConvention = CallingConvention.Cdecl)]
            public static extern void faiss_index_free(IntPtr index);

            [DllImport("FaissWrapper/Libs/win-x64/FaissWrapperNative.dll", CallingConvention = CallingConvention.Cdecl)]
            public static extern int faiss_index_add(IntPtr index, int n, float[] xb);

            [DllImport("FaissWrapper/Libs/win-x64/FaissWrapperNative.dll", CallingConvention = CallingConvention.Cdecl)]
            public static extern int faiss_index_search(IntPtr index, int n, float[] x, int k,
                                                     [Out] float[] distances, [Out] long[] labels);

            [DllImport("FaissWrapper/Libs/win-x64/FaissWrapperNative.dll", CallingConvention = CallingConvention.Cdecl)]
            public static extern IntPtr faiss_get_last_error();

            [DllImport("FaissWrapper/Libs/win-x64/FaissWrapperNative.dll", CallingConvention = CallingConvention.Cdecl)]
            public static extern int faiss_index_train(IntPtr index, int n, float[] xb);

            // Add this native method for checking training status
            [DllImport("FaissWrapper/Libs/win-x64/FaissWrapperNative.dll", CallingConvention = CallingConvention.Cdecl)]
            public static extern int faiss_index_is_trained(IntPtr index);
        }

        public FaissIndex(int dimensions = 384, FaissIndexDescription description=FaissIndexDescription.Flat)
        {
            _dimensions = dimensions;
            _description = description;  // Store the description
            int result = NativeMethods.faiss_index_factory(out _indexPtr, dimensions, Enum.GetName(description));
            if (result != 0) throw new FaissException($"Index creation failed: {FaissException.GetLastError()}");
        }
        public bool IsTrained
        {
            get
            {
                int result = NativeMethods.faiss_index_is_trained(_indexPtr);
                if (result == -1) throw new FaissException("Training check failed");
                return result == 1;
            }
        }

        public void Add(float[] vectors)
        {
            //if (!IsTrained && Description.Contains("IVF"))
            if (!IsTrained && Enum.GetName(_description).Contains("IVF"))
                throw new InvalidOperationException("IVF index must be trained before adding vectors");

            if (_disposed) throw new ObjectDisposedException("FaissIndex");
            if (vectors.Length % Dimensions != 0)
                throw new ArgumentException($"Vectors length must be multiple of index dimensions ({Dimensions})");

            int n = vectors.Length / Dimensions;
            int result = NativeMethods.faiss_index_add(_indexPtr, n, vectors);

            if (result != 0)
            {
                string errorDetails = FaissException.GetLastError();
                throw new FaissException($"Add failed (error {result}): {errorDetails}");
            }
        }

        public static string GetNativeError()
        {
            IntPtr ptr = NativeMethods.faiss_get_last_error();
            return ptr != IntPtr.Zero ? Marshal.PtrToStringAnsi(ptr) : "No error message available";
        }

        public (float[] Distances, long[] Labels) Search(string query, int NumberOfNearestNeighborsMatch, Onnx.EmbeddingMethods embeddingMethod=Onnx.EmbeddingMethods.CLS_Token)
        {
            float[] vector = Onnx.GetEmbedding(query, embeddingMethod);
            return Search(vector, NumberOfNearestNeighborsMatch);
        }
        public (float[] Distances, long[] Labels) Search(float[] queries, int NumberOfNearestNeighborsMatch)
        {
            if (_disposed) throw new ObjectDisposedException("FaissIndex");
            if (queries.Length % Dimensions != 0)
                throw new ArgumentException("Queries length must be multiple of index dimensions");

            int n = queries.Length / Dimensions;
            float[] distances = new float[n * NumberOfNearestNeighborsMatch];
            long[] labels = new long[n * NumberOfNearestNeighborsMatch];

            int result = NativeMethods.faiss_index_search(_indexPtr, n, queries, NumberOfNearestNeighborsMatch, distances, labels);
            if (result != 0) throw new FaissException($"Search failed (error {result})");

            return (distances, labels);
        }

        public List<SearchResult> GetDocumentsForSingleQuery(Dictionary<object, float[]> contents, long[] indices, float[] distances)
        {
            List<SearchResult> results = new List<SearchResult>();
            for (int i = 0; i < indices.Length; i++)
            {
                results.Add(new SearchResult() { Rank = i + 1, ContentID = contents.ElementAt((int)indices[i]).Key, Distance = distances[i] });
            }
            return results;
        }
        public Dictionary<string, List<SearchResult>> GetDocumentsForMultipleQueries(Dictionary<object, float[]> contents, int NumberOfNearestDocuments, long[] indices, float[] distances)
        {
            int k = NumberOfNearestDocuments;
            int numQueries = indices.Length / k;

            Dictionary<string, List<SearchResult>> Result = new Dictionary<string, List<SearchResult>>();

            List<SearchResult> searchResults = new List<SearchResult>();

            for (int q = 0; q < numQueries; q++)
            {
                for (int i = 0; i < k; i++)
                {
                    int idx = q * k + i;
                    int docIndex = (int)indices[idx];
                    float distance = distances[idx];

                    if (docIndex >= 0 && docIndex < contents.Count)
                    {
                        searchResults.Add(new SearchResult() { Rank = i + 1, ContentID = contents.ElementAt(docIndex).Key, Distance = distance });
                    }
                    else
                    {
                        searchResults.Add(new SearchResult() { Rank = i + 1, ContentID = -1, Distance = distance, ErrorMessage = $"Invalid document index: {docIndex}" });
                    }
                }
                Result.Add($"QUERY {q + 1}", searchResults);
            }

            return Result;
        }

        public void Train(float[] vectors)
        {
            if (Dimensions <= 0)
                throw new InvalidOperationException("Dimensions must be greater than zero");

            if (vectors == null)
                throw new ArgumentNullException(nameof(vectors));

            if (vectors.Length == 0)
                throw new ArgumentException("Training vectors cannot be empty", nameof(vectors));

            if (vectors.Length % Dimensions != 0)
                throw new ArgumentException($"Training vectors length ({vectors.Length}) must be a multiple of Dimensions ({Dimensions})");

            int n = vectors.Length / Dimensions;

            if (n <= 0) // Should not happen if previous checks pass, but just in case
                throw new InvalidOperationException("Number of vectors (n) must be positive");

            if (_indexPtr == IntPtr.Zero)
                throw new InvalidOperationException("FAISS index pointer is not initialized");

            int result = NativeMethods.faiss_index_train(_indexPtr, n, vectors);
            if (result != 0)
                throw new FaissException($"Training failed: {FaissException.GetLastError()}");
        }

        public void Dispose()
        {
            if (!_disposed && _indexPtr != IntPtr.Zero)
            {
                NativeMethods.faiss_index_free(_indexPtr);
                _indexPtr = IntPtr.Zero;
            }
            _disposed = true;
            GC.SuppressFinalize(this);
        }

        ~FaissIndex() => Dispose();

        public class FaissException : Exception
        {
            public FaissException(string message) : base(message) { }
            public static string GetLastError()
            {
                IntPtr ptr = NativeMethods.faiss_get_last_error();
                return ptr != IntPtr.Zero ? Marshal.PtrToStringAnsi(ptr) : "No error details available";
            }
        }
    }
}
