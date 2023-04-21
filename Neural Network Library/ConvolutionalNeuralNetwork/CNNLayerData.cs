using Neural_Network_Library.Core;

namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    internal struct CNNLayerData
    {
        internal readonly int _nKernels;
        internal readonly int _kernelSize;
        internal readonly ActivationFunctionType _activationFunctionType;

        internal CNNLayerData(int nKernels, int kernelSize, ActivationFunctionType activationFunctionType)
        {
            _nKernels = nKernels;
            _kernelSize = kernelSize;
            _activationFunctionType = activationFunctionType;
        }
    }
}