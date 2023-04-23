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

        public float[] FeedForwardTest(float[,] input)
        {
            long t = DateTime.Now.Ticks;

            Core.Debugging.ImageDrawer imageDrawer = new Core.Debugging.ImageDrawer(10, @"C:\Users\gabri\Desktop\Code Shit\TestFolder\");

            Tensor output = new Tensor(input);
            imageDrawer.SaveFloatMatrixAsBitmap($"{t} Input", input);

            for (int i = 0; i < _layers.Length; i++)
            {
                ConvolutionalLayer layer = _layers[i];
                output = layer.FeedForward(output);

                for (int d = 0; d < output.Depth; d++)
                {
                    imageDrawer.SaveFloatMatrixAsBitmap($"{t} Layer {i} - Kernel {d} output", output[d]);
                }
            }

            float[] FCLInput = output.Flatten();
            float[] FCLOutput = _fullyConnectedLayer.FeedForward(FCLInput);

            return FCLOutput;
        }
    }
}