using Neural_Network_Library.Core;
using Neural_Network_Library.Networks.MultilayeredPerceptron;
using System.Text;
using Random = Neural_Network_Library.Core.Random;

namespace Neural_Network_Library.SaveLoad
{
    public class MultilayeredPerceptronSaver
    {
        private readonly NeuralNetwork _network;

        public MultilayeredPerceptronSaver(NeuralNetwork network)
        {
            _network = network;
        }

        private string GetNetworkString()
        {
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < _network._layers.Length; i++)
            {
                sb.Append($"Layer {i}\n");
                sb.Append("[\n");

                sb.Append("\tWeights:\n\t[\n");
                for (int j = 0; j < _network._layers[i]._w.GetLength(0); j++)
                {
                    sb.Append("\t\t");
                    for (int k = 0; k < _network._layers[i]._w.GetLength(1); k++)
                    {
                        sb.Append($"{_network._layers[i]._w[j, k]} ");
                    }
                    sb.Append('\n');
                }

                sb.Append("\t]\n\tBiases:\n\t[\n");
                for (int j = 0; j < _network._layers[i]._b.Length; j++)
                {
                    sb.Append($"\t\t{_network._layers[i]._b[j]}\n");
                }

                sb.Append("\t]\n]\n");
            }

            return sb.ToString();
        }

        public void SaveNetwork()
        {
            string networkString = GetNetworkString();
            File.WriteAllText(@$"C:\Users\gabri\Desktop\Code Shit\Network{Random.Range(2000)}.txt", networkString);
        }
    }
}
