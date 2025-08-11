
//using Microsoft.Extensions.Configuration.FileExtensions;

using Microsoft.Extensions.Configuration;
using System;
using System.IO;
using System.Runtime.InteropServices;

namespace BanyanFaiss
{
    public static class Config
    {
        public static IConfiguration Configuration;

        public static string Tokenizer { get { return Configuration.GetSection("ONNX")["Tokenizer"]; } }
        public static string OnnxMiniLM { get { return Configuration.GetSection("ONNX")["OnnxMiniLM"]; } }
    }
}
