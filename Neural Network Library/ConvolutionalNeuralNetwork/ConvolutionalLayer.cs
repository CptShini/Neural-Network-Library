using Neural_Network_Library.Core;

namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    internal class ConvolutionalLayer
    {
        private readonly int _kernelDepth;
        private readonly int _nKernels;
        private readonly int _kernelSize;
        private readonly ActivationFunctionType _activationFunctionType;

        private readonly int _poolSize;

        private readonly Kernel[] _kernels;

        internal ConvolutionalLayer(int kernelDepth, CNNLayerData layerData)
        {
            _kernelDepth = kernelDepth;
            _nKernels = layerData._nKernels;
            _kernelSize = layerData._kernelSize;
            _activationFunctionType = layerData._activationFunctionType;

            _poolSize = 2;

            _kernels = new Kernel[layerData._nKernels];

            InitializeKernels();
        }

        private void InitializeKernels()
        {
            for (int i = 0; i < _nKernels; i++)
            {
                _kernels[i] = new Kernel(_kernelDepth, _kernelSize, _activationFunctionType);
            }
        }

        internal float[][,] FeedForward(float[][,] input)
        {
            float[][,] outputs = new float[_nKernels][,];

            for (int i = 0; i < _nKernels; i++)
            {
                outputs[i] = _kernels[i].Convolve(input);
                outputs[i] = MaxPool(outputs[i]);
            }

            return outputs;
        }

        private float[,] MaxPool(float[,] input)
        {
            int inputSize = input.GetLength(0);

            int maxPoolSize = inputSize / _poolSize;
            float[,] maxPool = new float[maxPoolSize, maxPoolSize];

            for (int x = 0; x < maxPoolSize; x++)
            {
                for (int y = 0; y < maxPoolSize; y++)
                {
                    maxPool[x, y] = MaxPoolAt(input, x, y);
                }
            }

            return maxPool;
        }

        private float MaxPoolAt(float[,] input, int x, int y)
        {
            int inputX = x * _poolSize;
            int inputY = y * _poolSize;

            float currentMax = 0f;
            for (int i = 0; i < _poolSize; i++)
            {
                for (int j = 0; j < _poolSize; j++)
                {
                    int offsetX = i;
                    int offsetY = j;

                    int offsetInputX = inputX + offsetX;
                    int offsetInputY = inputY + offsetY;

                    float val = input[offsetInputX, offsetInputY];
                    if (val > currentMax) currentMax = val;
                }
            }

            return currentMax;
        }
    }
}