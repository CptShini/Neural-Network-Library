using Neural_Network_Library.Core;

namespace Neural_Network_Library.Networks.ConvolutionalNeuralNetwork
{
    public class CNNStructure
    {
        private readonly List<CNNLayerData> _layerData;

        public CNNStructure() => _layerData = new List<CNNLayerData>();

        public void AddLayer(int nKernels, int kernelSize, ActivationFunctionType activationFunctionType) => _layerData.Add(new CNNLayerData(nKernels, kernelSize, activationFunctionType));

        internal ConvolutionalLayer[] GetLayerStructure()
        {
            List<ConvolutionalLayer> layers = new List<ConvolutionalLayer> { new ConvolutionalLayer(1, _layerData[0]) };

            for (int i = 1; i < _layerData.Count; i++)
            {
                ConvolutionalLayer convolutionalLayer = new ConvolutionalLayer(_layerData[i - 1].nKernels, _layerData[i]);
                layers.Add(convolutionalLayer);
            }

            return layers.ToArray();
        }

        internal int GetFCLInputSize(int inputSize)
        {
            int finalCNNLayerOutputSize = inputSize;
            foreach (CNNLayerData layer in _layerData)
            {
                finalCNNLayerOutputSize = (finalCNNLayerOutputSize - (layer.kernelSize - 1)) / 2;
            }

            int outputSize = finalCNNLayerOutputSize * finalCNNLayerOutputSize * _layerData[^1].nKernels;
            return outputSize;
        }
    }

    internal record struct CNNLayerData(int nKernels, int kernelSize, ActivationFunctionType activationFunctionType);
}