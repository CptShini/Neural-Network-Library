namespace Neural_Network_Library
{
    public class NeuralNetwork
    {
        internal readonly Layer[] layers;

        public NeuralNetwork(int[] networkStructure, ActivationFunctionType activationFunction = ActivationFunctionType.Sigmoid)
        {
            layers = new Layer[networkStructure.Length - 1];

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(networkStructure[i], networkStructure[i + 1], activationFunction);
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
    }
}