using Neural_Network_Library.Core;

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
            float[][,] output = new float[][,] { input };
            foreach (ConvolutionalLayer layer in _layers)
            {
                output = layer.FeedForward(output);
            }

            float[] FCLInput = Flatten(output);
            float[] FCLOutput = _fullyConnectedLayer.FeedForward(FCLInput);

            return FCLOutput;
        }

        private static float[] Flatten(float[][,] input)
        {
            int inputSize = input[0].GetLength(0);
            int outputLength = input.Length * inputSize * inputSize;
            float[] output = new float[outputLength];

            int n = 0;
            for (int i = 0; i < input.Length; i++)
            {
                for (int x = 0; x < inputSize; x++)
                {
                    for (int y = 0; y < inputSize; y++)
                    {
                        output[n++] = input[i][x, y];
                    }
                }
            }

            return output;
        }
    }
}