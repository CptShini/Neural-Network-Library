using Neural_Network_Library.Core;
using Neural_Network_Library.Interfaces.CNN;
using Neural_Network_Library.Networks.MLP;

namespace Neural_Network_Library.Networks.CNN;

public class ConvolutionalNeuralNetwork : IConvolutionalNeuralNetwork
{
    internal readonly ConvolutionalLayer[] _convolutionalLayers;
    internal readonly MultilayeredPerceptron _fullyConnectedLayer;

    public ConvolutionalNeuralNetwork(CNNStructure networkStructure)
    {
        _convolutionalLayers = networkStructure.BuildLayers();
        int FCLInputSize = networkStructure.GetFCLInputSize();

        MLPStructure FCLStructure = new MLPStructure(FCLInputSize);
        FCLStructure.AddLayer(networkStructure._outputSize, ActivationFunctionType.ReLU);

        _fullyConnectedLayer = new MultilayeredPerceptron(FCLStructure);
    }

    public float[] FeedForward(float[,] input)
    {
        Tensor output = new Tensor(input);
        foreach (IConvolutionalLayer convolutionalLayer in _convolutionalLayers)
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