using Neural_Network_Library.Interfaces.MLP;

namespace Neural_Network_Library.Networks.MLP;

public class MultilayeredPerceptron : INeuralNetwork
{
    internal readonly MLPLayer[] _layers;

    public MultilayeredPerceptron(MLPStructure layerStructure) => _layers = layerStructure.BuildLayers();

    public float[] FeedForward(float[] input)
    {
        foreach (ILayer layer in _layers)
        {
            input = layer.FeedForward(input);
        }

        return input;
    }
}