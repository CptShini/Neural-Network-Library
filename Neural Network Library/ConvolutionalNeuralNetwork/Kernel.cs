using static Neural_Network_Library.NeuralNetworkMath;

namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    internal class Kernel
    {
        private readonly int _nFilters;
        private readonly int _kernelSize;
        private readonly int _kernelSizeDelta;
        private readonly ActivationFunctionType _activationFunctionType;

        private readonly float[][,] _filter;
        private float _bias;

        internal Kernel(int kernelSize, int nFilters, ActivationFunctionType activationFunctionType)
        {
            _nFilters = nFilters;
            _kernelSize = kernelSize;
            _kernelSizeDelta = (int)(kernelSize / 2f - 0.5f);
            _activationFunctionType = activationFunctionType;

            _filter = new float[nFilters][,];
            InitializeFiltersAndBias();
        }

        private void InitializeFiltersAndBias()
        {
            for (int i = 0; i < _nFilters; i++)
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

        internal float[,] ApplyFilters(float[][,] input)
        {
            int inputSize = input[0].GetLength(0);

            int convolutionSize = inputSize - 2 * _kernelSizeDelta;
            float[,] convolution = new float[convolutionSize, convolutionSize];

            for (int x = 0; x < convolutionSize; x++)
            {
                for (int y = 0; y < convolutionSize; y++)
                {
                    int cellX = x + _kernelSizeDelta;
                    int cellY = y + _kernelSizeDelta;

                    convolution[x, y] = ConvolveAt(input, cellX, cellY);
                }
            }

            return convolution;
        }

        private float ConvolveAt(float[][,] input, int cellX, int cellY)
        {
            float convolutionSum = 0f;

            for (int i = 0; i < _nFilters; i++)
            {
                convolutionSum += ConvolveAtForSpecific(input[i], cellX, cellY, _filter[i]);
            }

            float z = convolutionSum + _bias;
            float a = ActivationFunction.Activate(z, _activationFunctionType);

            return a;
        }

        private float ConvolveAtForSpecific(float[,] input, int cellX, int cellY, float[,] _filter)
        {
            float convolutionSum = 0f;

            for (int i = 0; i < _kernelSize; i++)
            {
                for (int j = 0; j < _kernelSize; j++)
                {
                    int offsetX = i - _kernelSizeDelta;
                    int offsetY = j - _kernelSizeDelta;

                    int offsetCellPosX = cellX + offsetX;
                    int offsetCellPosY = cellY + offsetY;

                    convolutionSum += input[offsetCellPosX, offsetCellPosY] * _filter[i, j];
                }
            }

            return convolutionSum;
        }
    }
}