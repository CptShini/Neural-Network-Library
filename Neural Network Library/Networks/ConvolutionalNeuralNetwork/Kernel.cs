using Neural_Network_Library.Core;
using static Neural_Network_Library.Core.ActivationFunction;
using static Neural_Network_Library.Core.RandomNumberGenerator;

namespace Neural_Network_Library.Networks.ConvolutionalNeuralNetwork
{
    internal class Kernel : IKernel
    {
        internal Tensor _filter;
        internal float _bias;

        internal readonly ActivationFunctionType _activationFunctionType;

        internal Kernel(int kernelDepth, int kernelSize, ActivationFunctionType activationFunctionType)
        {
            _activationFunctionType = activationFunctionType;

            _filter = new Tensor(kernelDepth, kernelSize);
            InitializeFiltersAndBias();
        }

        private void InitializeFiltersAndBias()
        {
            for (int d = 0; d < _filter.Depth; d++)
            {
                for (int x = 0; x < _filter.Size; x++)
                {
                    for (int y = 0; y < _filter.Size; y++)
                    {
                        _filter[d, x, y] = RandomRange(-1f, 1f);
                    }
                }
            }

            _bias = RandomRange(-1f, 1f);
        }

        float[,] IKernel.Convolve(Tensor input)
        {
            int convolutionSize = input.Size - (_filter.Size - 1);
            float[,] convolution = new float[convolutionSize, convolutionSize];

            for (int x = 0; x < convolutionSize; x++)
            {
                for (int y = 0; y < convolutionSize; y++)
                {
                    convolution[x, y] = _bias;

                    for (int d = 0; d < input.Depth; d++)
                    {
                        for (int i = 0; i < _filter.Size; i++)
                        {
                            for (int j = 0; j < _filter.Size; j++)
                            {
                                convolution[x, y] += input[d, x + i, y + j] * _filter[d, i, j];
                            }
                        }
                    }

                    convolution[x, y] = convolution[x, y].Activate(_activationFunctionType);
                }
            }

            return convolution;
        }
    }
}