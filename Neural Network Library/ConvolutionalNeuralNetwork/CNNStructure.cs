using Neural_Network_Library.Core;

namespace Neural_Network_Library.ConvolutionalNeuralNetwork
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
                layers.Add(new ConvolutionalLayer(_layerData[i - 1]._nKernels, _layerData[i]));
            }

            return layers.ToArray();
        }

        internal int GetFCLayerInputSize(int inputSize)
        {
            int finalCNNLayerOutputSize = inputSize;
            foreach (CNNLayerData layer in _layerData)
            {
                finalCNNLayerOutputSize = (inputSize - (layer._kernelSize - 1)) / 2;
            }

            int outputSize = finalCNNLayerOutputSize * _layerData[^1]._nKernels;
            return outputSize;
        }
    }
}