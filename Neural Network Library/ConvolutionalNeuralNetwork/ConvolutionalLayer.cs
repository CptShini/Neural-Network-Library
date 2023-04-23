using Neural_Network_Library.Core.Math;

namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    internal class ConvolutionalLayer
    {
        private readonly Kernel[] _kernels;

        internal ConvolutionalLayer(int inputDepth, CNNLayerData layerData)
        {
            _kernels = new Kernel[layerData.nKernels];
            for (int i = 0; i < _kernels.Length; i++)
            {
                _kernels[i] = new Kernel(inputDepth, layerData.kernelSize, layerData.activationFunctionType);
            }
        }

        internal Tensor FeedForward(Tensor input) => _kernels.Convolve(input);
    }
}