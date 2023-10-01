namespace Neural_Network_Library.Networks.ConvolutionalNeuralNetwork
{
    internal class ConvolutionalLayer
    {
        private readonly int _kernelCount;
        private readonly Kernel[] _kernels;

        internal ConvolutionalLayer(int inputDepth, CNNLayerData layerData)
        {
            _kernelCount = layerData.nKernels;

            _kernels = new Kernel[_kernelCount];
            for (int i = 0; i < _kernelCount; i++)
            {
                _kernels[i] = new Kernel(inputDepth, layerData.kernelSize, layerData.activationFunctionType);
            }
        }

        internal Tensor FeedForward(Tensor input)
        {
            Tensor output = new Tensor(_kernelCount);

            for (int d = 0; d < _kernelCount; d++)
            {
                output[d] = _kernels[d].Convolve(input);
                output[d] = MaxPool(output[d]);
            }

            return output;
        }

        private static float[,] MaxPool(float[,] input)
        {
            int poolSize = 2;
            int inputSize = input.GetLength(0);

            int maxPoolSize = inputSize / poolSize;
            float[,] maxPool = new float[maxPoolSize, maxPoolSize];

            for (int x = 0; x < maxPoolSize; x++)
            {
                for (int y = 0; y < maxPoolSize; y++)
                {
                    int inputX = x * poolSize;
                    int inputY = y * poolSize;

                    for (int i = 0; i < poolSize; i++)
                    {
                        for (int j = 0; j < poolSize; j++)
                        {
                            int offsetInputX = inputX + i;
                            int offsetInputY = inputY + j;

                            float val = input[offsetInputX, offsetInputY];
                            if (val > maxPool[x, y]) maxPool[x, y] = val;
                        }
                    }
                }
            }

            return maxPool;
        }
    }
}