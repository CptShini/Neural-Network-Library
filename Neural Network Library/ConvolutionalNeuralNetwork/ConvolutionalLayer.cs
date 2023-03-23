namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    internal class ConvolutionalLayer
    {
        private readonly int _depth;
        private readonly int _nFilters;
        private readonly int _kernelSize;
        private readonly ActivationFunctionType _activationFunctionType;

        private readonly Kernel[] _kernels;

        private readonly int _poolSize = 2;

        internal ConvolutionalLayer(int depth, int nFilters, int kernelSize, ActivationFunctionType activationType)
        {
            _depth = depth;
            _nFilters = nFilters;
            _kernelSize = kernelSize;
            _activationFunctionType = activationType;
            _kernels = new Kernel[nFilters];

            InitializeKernels();
        }

        private void InitializeKernels()
        {
            for (int i = 0; i < _nFilters; i++)
            {
                _kernels[i] = new Kernel(_kernelSize, _depth, _activationFunctionType);
            }
        }

        internal float[][,] Process(float[][,] input)
        {
            float[][,] outputs = new float[_nFilters][,];

            for (int i = 0; i < _nFilters; i++)
            {
                outputs[i] = _kernels[i].ApplyFilters(input);
                outputs[i] = MaxPool(outputs[i]);
            }

            return outputs;
        }

        private float[,] MaxPool(float[,] input)
        {
            int inputSize = input.GetLength(0);

            int maxPoolSize = inputSize / _poolSize;
            float[,] maxPool = new float[maxPoolSize, maxPoolSize];

            for (int i = 0; i < maxPoolSize; i++)
            {
                for (int j = 0; j < maxPoolSize; j++)
                {
                    int cellX = i * _poolSize;
                    int cellY = j * _poolSize;
                    maxPool[i, j] = MaxPoolAt(input, cellX, cellY);
                }
            }

            return maxPool;
        }

        private float MaxPoolAt(float[,] input, int cellX, int cellY)
        {
            float currentMax = 0f;

            for (int i = 0; i < _poolSize; i++)
            {
                for (int j = 0; j < _poolSize; j++)
                {
                    int offsetCellPosX = cellX + i;
                    int offsetCellPosY = cellY + j;

                    if (input[offsetCellPosX, offsetCellPosY] > currentMax) currentMax = input[offsetCellPosX, offsetCellPosY];
                }
            }

            return currentMax;
        }

        //size of filter
        //use zero padding - bool
        //valid - no padding
        //same - maintain input size padding
        //Enum?

        //convl layers, n filters, filter = kernel, filter dimensions, randomize filter initialization
    }
}
