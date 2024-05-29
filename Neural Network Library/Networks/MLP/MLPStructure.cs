using Neural_Network_Library.Core;

namespace Neural_Network_Library.Networks.MLP;

public class MLPStructure
{
    private record MLPLayerData(int outputSize, ActivationFunctionType activationFunctionType);

    private readonly List<MLPLayerData> _layerData;
    private readonly int _inputSize;

    public MLPStructure(int inputSize)
    {
        if (inputSize < 1) throw new ArgumentOutOfRangeException($"The given input size, {inputSize}, is zero or negative and is therefore not valid.");

        _inputSize = inputSize;
        _layerData = new List<MLPLayerData>();
    }

    public void AddLayer(int outputSize, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
    {
        if (outputSize < 1) throw new ArgumentOutOfRangeException($"The given output size, {outputSize}, is zero or negative and is therefore not valid.");

        MLPLayerData layerData = new MLPLayerData(outputSize, activationFunctionType);
        _layerData.Add(layerData);
    }
    
    internal MLPLayer[] BuildLayers()
    {
        List<MLPLayer> layers = new List<MLPLayer>() { BuildLayer(_inputSize, _layerData[0]) };
        for (int i = 1; i < _layerData.Count; i++)
        {
            MLPLayer layer = BuildLayer(_layerData[i - 1].outputSize, _layerData[i]);
            layers.Add(layer);
        }

        return layers.ToArray();
    }

    private static MLPLayer BuildLayer(int inputSize, MLPLayerData layerData) => new(inputSize, layerData.outputSize, layerData.activationFunctionType);
}