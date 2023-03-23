namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    public class ConvolutionalNeuralNetwork
    {
        private readonly List<int> _layerFilters;
        private readonly List<int> _layerKernelSizes;
        private readonly List<ActivationFunctionType> _layerActivationFunctionTypes;

        private readonly List<ConvolutionalLayer> _layers;

        private bool _initialized = false;

        public ConvolutionalNeuralNetwork()
        {
            _layers = new List<ConvolutionalLayer>();
            _layerFilters = new List<int>();
            _layerKernelSizes = new List<int>();
            _layerActivationFunctionTypes = new List<ActivationFunctionType>();
        }

        public void AddLayer(int nFilters, int kernelSize, ActivationFunctionType activationFunctionType)
        {
            _layerFilters.Add(nFilters);
            _layerKernelSizes.Add(kernelSize);
            _layerActivationFunctionTypes.Add(activationFunctionType);
        }

        public void InitializeConvolutionalNeuralNetwork()
        {
            ConvolutionalLayer layer = new ConvolutionalLayer(1, _layerFilters[0], _layerKernelSizes[0], _layerActivationFunctionTypes[0]);
            _layers.Add(layer);

            for (int i = 1; i < _layerFilters.Count; i++)
            {
                layer = new ConvolutionalLayer(_layerFilters[i - 1], _layerFilters[i], _layerKernelSizes[i], _layerActivationFunctionTypes[i]);
                _layers.Add(layer);
            }

            _initialized = true;
        }

        public float[][,] GetOutputs(float[,] input)
        {
            if (!_initialized) return null;

            float[][,] output = new float[][,] { input };

            foreach (ConvolutionalLayer layer in _layers)
            {
                output = layer.Process(output);
            }

            return output;
        }
    }
}
