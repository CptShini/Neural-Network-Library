using Neural_Network_Library.Core;
using Neural_Network_Library.Core.Math;

namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    public class ConvolutionalNeuralNetwork
    {
        private readonly ConvolutionalLayer[] _layers;
        private readonly MultilayeredPerceptron.NeuralNetwork _fullyConnectedLayer;

        public ConvolutionalNeuralNetwork(int inputSize, int outputLength, CNNStructure networkStructure)
        {
            _layers = networkStructure.GetLayerStructure();
            int FCLInputSize = networkStructure.GetFCLInputSize(inputSize);

            int[] FCLStructure = { FCLInputSize , outputLength };
            _fullyConnectedLayer = new MultilayeredPerceptron.NeuralNetwork(FCLStructure, ActivationFunctionType.ReLU);
        }

        public float[] FeedForward(float[,] input)
        {
            Tensor output = new Tensor(input);
            foreach (ConvolutionalLayer layer in _layers)
            {
                output = layer.FeedForward(output);
            }

            float[] FCLInput = output.Flatten();
            float[] FCLOutput = _fullyConnectedLayer.FeedForward(FCLInput);

            return FCLOutput;
        }

        private static float[] Flatten(Tensor input)
        {
            int inputSize = input.Size;
            int outputLength = input.Volume;
            float[] output = new float[outputLength];

            int n = 0;
            for (int i = 0; i < input.Depth; i++)
            {
                for (int x = 0; x < inputSize; x++)
                {
                    for (int y = 0; y < inputSize; y++)
                    {
                        output[n++] = input[i, x, y];
                    }
                }
            }

            return output;
        }
    }
}