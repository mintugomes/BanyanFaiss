using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace BanyanFaiss
{
    internal class TokenizerConfig
    {
        public Dictionary<string, int> Vocab { get; private set; }

        public TokenizerConfig(string tokenizerJsonPath)
        {
            var json = JObject.Parse(File.ReadAllText(tokenizerJsonPath));
            var vocabJson = json["model"]["vocab"] as JObject;

            Vocab = vocabJson.Properties()
                             .ToDictionary(p => p.Name, p => (int)p.Value);
        }

        public int GetId(string token) => Vocab.ContainsKey(token) ? Vocab[token] : Vocab["[UNK]"];
        public bool Contains(string token) => Vocab.ContainsKey(token);
    }
}
