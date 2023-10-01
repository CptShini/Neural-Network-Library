using Neural_Network_Library.Core;

namespace Neural_Network_Library.Networks.ConvolutionalNeuralNetwork
{
    public class ConvolutionalNeuralNetwork
    {
        private readonly ConvolutionalLayer[] _convolutionalLayers;
        private readonly MultilayeredPerceptron.NeuralNetwork _fullyConnectedLayer;

        public ConvolutionalNeuralNetwork(int inputSize, int outputLength, CNNStructure networkStructure)
        {
            _convolutionalLayers = networkStructure.GetLayerStructure();

            int FCLInputSize = networkStructure.GetFCLInputSize(inputSize);
            int[] FCLStructure = { FCLInputSize , outputLength };
            _fullyConnectedLayer = new MultilayeredPerceptron.NeuralNetwork(FCLStructure, ActivationFunctionType.ReLU);
        }

        public float[] FeedForward(float[,] input)
        {
            Tensor output = new Tensor(input);
            foreach (ConvolutionalLayer convolutionalLayer in _convolutionalLayers)
            {
                output = convolutionalLayer.FeedForward(output);
            }

            float[] FCLInput = Flatten(output);
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