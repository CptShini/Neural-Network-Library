using Neural_Network_Library.Core;

namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    internal class ConvolutionalLayer
    {
        private readonly int _kernelDepth;
        private readonly int _kernelSize;
        private readonly ActivationFunctionType _activationFunctionType;

        private readonly int _poolSize;

        private readonly Kernel[] _kernels;

        internal ConvolutionalLayer(int kernelDepth, CNNLayerData layerData)
        {
            _kernelDepth = kernelDepth;
            _kernelSize = layerData.kernelSize;
            _activationFunctionType = layerData.activationFunctionType;

            _poolSize = 2;

            _kernels = new Kernel[layerData.nKernels];
            InitializeKernels();
        }

        private void InitializeKernels()
        {
            for (int i = 0; i < _kernels.Length; i++)
            {
                _kernels[i] = new Kernel(_kernelDepth, _kernelSize, _activationFunctionType);
            }
        }

        internal Tensor FeedForward(Tensor input)
        {
            Tensor output = new Tensor(_kernels.Length);

            for (int d = 0; d < output.Depth; d++)
            {
                output[d] = _kernels[d].Convolve(input).MaxPool(_poolSize);
            }

            return output;
        }
    }
}