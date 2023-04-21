using Neural_Network_Library.Core;

namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    public class ConvolutionalNeuralNetwork
    {
        private readonly ConvolutionalLayer[] _layers;
        private readonly MultilayeredPerceptron.NeuralNetwork _fullyConnectedLayer;

        private readonly int _fullyConnectedLayerInputSize;

        public ConvolutionalNeuralNetwork(int inputSize, int outputSize, CNNStructure networkStructure)
        {
            _layers = networkStructure.GetLayerStructure();
            _fullyConnectedLayerInputSize = networkStructure.GetFCLayerInputSize(inputSize);

            int[] FCStructure = { _fullyConnectedLayerInputSize , outputSize };
            _fullyConnectedLayer = new MultilayeredPerceptron.NeuralNetwork(FCStructure, ActivationFunctionType.ReLU);
        }

        public float[] FeedForward(float[,] input)
        {
            float[][,] output = new float[][,] { input };

            foreach (ConvolutionalLayer layer in _layers)
            {
                output = layer.FeedForward(output);
            }

            float[] FCInput = NeuralNetworkMath.Flatten(output);
            float[] FCOutput = _fullyConnectedLayer.FeedForward(FCInput);

            return FCOutput;
        }
    }
}