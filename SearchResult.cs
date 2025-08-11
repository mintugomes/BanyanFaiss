using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BanyanFaiss
{
    public class SearchResult
    {
        public int Rank {  get; set; }
        public object ContentID {  get; set; }
        public float Distance {  get; set; }
        public string? ErrorMessage { get; set; }
    }
}
