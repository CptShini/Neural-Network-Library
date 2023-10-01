using Neural_Network_Library.Core;

namespace Neural_Network_Library.Networks.MultilayeredPerceptron
{
    public class NeuralNetwork
    {
        internal readonly Layer[] _layers;

        public NeuralNetwork(int[] networkStructure, ActivationFunctionType activationFunction = ActivationFunctionType.Sigmoid)
        {
            int layerCount = networkStructure.Length - 1;

            _layers = new Layer[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                _layers[i] = new Layer(networkStructure[i], networkStructure[i + 1], activationFunction);
            }
        }

        public float[] FeedForward(float[] input)
        {
            foreach (Layer layer in _layers)
            {
                input = layer.FeedForward(input);
            }

            return input;
        }
    }
}