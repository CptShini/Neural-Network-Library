namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    public class CNNStructure
    {
        private readonly List<int> _layerKernels;
        private readonly List<int> _layerKernelSizes;
        private readonly List<ActivationFunctionType> _layerActivationFunctionTypes;

        public CNNStructure()
        {
            _layerKernels = new List<int>();
            _layerKernelSizes = new List<int>();
            _layerActivationFunctionTypes = new List<ActivationFunctionType>();
        }

        public void AddLayer(int nKernels, int kernelSize, ActivationFunctionType activationFunctionType)
        {
            _layerKernels.Add(nKernels);
            _layerKernelSizes.Add(kernelSize);
            _layerActivationFunctionTypes.Add(activationFunctionType);
        }

        internal ConvolutionalLayer[] GetLayerStructure()
        {
            List<ConvolutionalLayer> layers = new List<ConvolutionalLayer>();

            ConvolutionalLayer layer = new ConvolutionalLayer(1, _layerKernels[0], _layerKernelSizes[0], _layerActivationFunctionTypes[0]);
            layers.Add(layer);

            for (int i = 1; i < _layerKernels.Count; i++)
            {
                layer = new ConvolutionalLayer(_layerKernels[i - 1], _layerKernels[i], _layerKernelSizes[i], _layerActivationFunctionTypes[i]);
                layers.Add(layer);
            }

            return layers.ToArray();
        }
    }
}
