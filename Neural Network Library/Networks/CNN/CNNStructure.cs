using Neural_Network_Library.Core;

namespace Neural_Network_Library.Networks.CNN;

public class CNNStructure
{
    private record CNNLayerData(int nKernels, int kernelSize, ActivationFunctionType activationFunctionType);

    private readonly List<CNNLayerData> _layerData;
    private readonly int _inputSize;
    internal readonly int _outputSize;

    public CNNStructure(int inputSize, int outputSize)
    {
        if (inputSize < 1) throw new ArgumentOutOfRangeException($"The given input size, {inputSize}, is zero or negative and is therefore not valid.");
        _inputSize = inputSize;

        if (outputSize < 1) throw new ArgumentOutOfRangeException($"The given output size, {outputSize}, is zero or negative and is therefore not valid.");
        _outputSize = outputSize;

        _layerData = new List<CNNLayerData>();
    }

    public void AddLayer(int nKernels, int kernelSize, ActivationFunctionType activationFunctionType)
    {
        if (nKernels < 1) throw new ArgumentOutOfRangeException($"The given number of kernels, {nKernels}, is zero or negative and is therefore not valid.");
        if (kernelSize < 1) throw new ArgumentOutOfRangeException($"The given kernel size, {kernelSize}, is zero or negative and is therefore not valid.");

        CNNLayerData layerData = new CNNLayerData(nKernels, kernelSize, activationFunctionType);
        _layerData.Add(layerData);
    }

    internal ConvolutionalLayer[] BuildLayers()
    {
        List<ConvolutionalLayer> layers = new List<ConvolutionalLayer> { BuildLayer(1, _layerData[0]) };
        for (int i = 1; i < _layerData.Count; i++)
        {
            ConvolutionalLayer convolutionalLayer = BuildLayer(_layerData[i - 1].nKernels, _layerData[i]);
            layers.Add(convolutionalLayer);
        }

        return layers.ToArray();
    }

    private static ConvolutionalLayer BuildLayer(int inputDepth, CNNLayerData layerData) => new(inputDepth, layerData.nKernels, layerData.kernelSize, layerData.activationFunctionType);

    internal int GetFCLInputSize()
    {
        int finalCNNLayerOutputSize = _inputSize;
        foreach (CNNLayerData layer in _layerData)
        {
            finalCNNLayerOutputSize -= layer.kernelSize - 1;
            finalCNNLayerOutputSize /= 2;
        }

        int outputSize = finalCNNLayerOutputSize * finalCNNLayerOutputSize * _layerData[^1].nKernels;
        return outputSize;
    }
}