using Neural_Network_Library.Core;
using static Neural_Network_Library.Core.ActivationFunction;
using Random = Neural_Network_Library.Core.Random;

namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    internal class Kernel
    {
        private readonly int _depth;
        private readonly int _kernelSize;
        private readonly int _kernelSizeDelta;
        private readonly ActivationFunctionType _activationFunctionType;

        private readonly float[][,] _filter;
        private float _bias;

        private float[][,] _input;

        internal Kernel(int depth, int kernelSize, ActivationFunctionType activationFunctionType)
        {
            _depth = depth;
            _kernelSize = kernelSize;
            _kernelSizeDelta = (int)((kernelSize - 1) / 2f);
            _activationFunctionType = activationFunctionType;

            _filter = new float[depth][,];
            InitializeFiltersAndBias();

            _input = new float[depth][,];
        }

        private void InitializeFiltersAndBias()
        {
            for (int i = 0; i < _depth; i++)
            {
                _filter[i] = new float[_kernelSize, _kernelSize];
                for (int x = 0; x < _kernelSize; x++)
                {
                    for (int y = 0; y < _kernelSize; y++)
                    {
                        _filter[i][x, y] = Random.Range(-1f, 1f);
                    }
                }
            }

            _bias = Random.Range(-1f, 1f);
        }

        internal float[,] Convolve(float[][,] input)
        {
            _input = input;
            int inputSize = input[0].GetLength(0);

            int convolutionSize = inputSize - 2 * _kernelSizeDelta;
            float[,] convolution = new float[convolutionSize, convolutionSize];

            for (int x = 0; x < convolutionSize; x++)
            {
                for (int y = 0; y < convolutionSize; y++)
                {
                    convolution[x, y] = ApplyFilters(x, y);
                }
            }

            return convolution;
        }

        private float ApplyFilters(int x, int y)
        {
            int inputX = x + _kernelSizeDelta;
            int inputY = y + _kernelSizeDelta;

            float convolutionSum = 0f;
            for (int i = 0; i < _depth; i++)
            {
                convolutionSum += ConvolveAtDepth(i, inputX, inputY);
            }

            float z = convolutionSum + _bias;
            float a = Activate(z, _activationFunctionType);

            return a;
        }

        private float ConvolveAtDepth(int depthIndex, int inputX, int inputY)
        {
            float convolutionSum = 0f;

            for (int i = 0; i < _kernelSize; i++)
            {
                for (int j = 0; j < _kernelSize; j++)
                {
                    int offsetX = i - _kernelSizeDelta;
                    int offsetY = j - _kernelSizeDelta;

                    int offsetInputX = inputX + offsetX;
                    int offsetInputY = inputY + offsetY;

                    convolutionSum += _input[depthIndex][offsetInputX, offsetInputY] * _filter[depthIndex][i, j];
                }
            }

            return convolutionSum;
        }
    }
}