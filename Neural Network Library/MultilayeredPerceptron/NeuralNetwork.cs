namespace Neural_Network_Library.MultilayeredPerceptron
{
    public class NeuralNetwork
    {
        internal readonly Layer[] _layers;

        public NeuralNetwork(int[] networkStructure, ActivationFunctionType activationFunction = ActivationFunctionType.Sigmoid)
        {
            _layers = new Layer[networkStructure.Length - 1];

            for (int i = 0; i < _layers.Length; i++)
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