namespace Neural_Network_Library
{
    public class NeuralNetwork
    {
        internal readonly Layer[] layers;

        public NeuralNetwork(int[] networkStructure)
        {
            layers = new Layer[networkStructure.Length - 1];

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(networkStructure[i], networkStructure[i + 1]);
            }
        }

        public float[] FeedForward(float[] input)
        {
            foreach (Layer layer in layers)
            {
                input = layer.FeedForward(input);
            }

            return input;
        }

        public void PrintNetwork()
        {
            string result = "";

            result += "Neurons:\n";

            for (int j = 0; j < layers[0].a_1.Length; j++)
            {
                result += $"[{layers[0].a_1[j]:0.00}]";
            }
            result += "\n";

            for (int i = 0; i < layers.Length; i++)
            {
                for (int j = 0; j < layers[i].a.Length; j++)
                {
                    result += $"[{layers[i].a[j]:0.00}]";
                }
                result += "\n";
            }

            result += "\n\n";

            result += "Weights:\n";

            for (int i = 0; i < layers.Length; i++)
            {
                for (int j = 0; j < layers[i].w.GetLength(0); j++)
                {
                    for (int k = 0; k < layers[i].w.GetLength(1); k++)
                    {
                        result += $"[{layers[i].w[j, k]:0.00}]";
                    }
                    result += "\n";
                }
                result += "\n";
            }

            result += "Biases:\n";

            for (int i = 0; i < layers.Length; i++)
            {
                for (int j = 0; j < layers[i].b.Length; j++)
                {
                    result += $"[{layers[i].b[j]:0.00}]";
                }
                result += "\n";
            }

            Console.WriteLine(result);
        }
    }
}