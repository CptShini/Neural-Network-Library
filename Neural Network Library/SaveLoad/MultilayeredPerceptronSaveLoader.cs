using Neural_Network_Library.Core;
using Neural_Network_Library.Networks.MLP;
using System.Text;

namespace Neural_Network_Library.SaveLoad
{
    public class MultilayeredPerceptronSaveLoader : NetworkSaveLoader<MultilayeredPerceptron>
    {
        private readonly MultilayeredPerceptron _network;
        private readonly StringBuilder sb = new();

        public MultilayeredPerceptronSaveLoader(MultilayeredPerceptron network) => _network = network;

        private protected override string EncodeNetwork()
        {
            sb.Clear();

            for (int i = 0; i < _network._layers.Length; i++)
            {
                AddLayerString(_network._layers[i], i);
            }

            return sb.ToString();
        }

        private void AddLayerString(MLPLayer layer, int index)
        {
            sb.AppendLine($"Layer {index}");
            sb.Append('[');
            sb.AppendLine();

            AddWeightsString(layer._w);
            AddBiasesString(layer._b);
            AddActivationFunctionString(layer._activationFunctionType);

            sb.AppendLine("]\n");
        }

        private void AddWeightsString(float[,] w)
        {
            sb.AppendLine("\tWeights");
            sb.AppendLine("\t[");

            for (int j = 0; j < w.GetLength(0); j++)
            {
                sb.Append("\t\t");
                for (int k = 0; k < w.GetLength(1); k++)
                {
                    sb.Append(w[j, k]);
                    sb.Append(' ');
                }

                sb.AppendLine();
            }

            sb.AppendLine("\t]\n");
        }

        private void AddBiasesString(float[] b)
        {
            sb.AppendLine("\tBiases");
            sb.AppendLine("\t[");

            for (int j = 0; j < b.Length; j++)
            {
                sb.Append("\t\t");
                sb.Append(b[j]);
                sb.AppendLine();
            }

            sb.AppendLine("\t]\n");
        }

        private void AddActivationFunctionString(ActivationFunctionType activationFunctionType)
        {
            sb.AppendLine("\tActivation Function");

            sb.Append('\t');
            sb.AppendLine(activationFunctionType.ToString());

            sb.AppendLine();
        }

        }
}