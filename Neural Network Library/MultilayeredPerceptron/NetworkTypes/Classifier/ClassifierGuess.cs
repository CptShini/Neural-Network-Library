using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_Library.MultilayeredPerceptron.NetworkTypes.Classifier
{
    public class ClassifierGuess
    {
        public int GuessIndex;
        public float GuessConfidence;
        public float[]? Outputs;
    }
}
