namespace Neural_Network_Library.Networks.ConvolutionalNeuralNetwork
{
    internal class ConvolutionalLayer : IConvolutionalLayer
    {
        internal readonly Kernel[] _kernels;

        internal ConvolutionalLayer(int inputDepth, CNNLayerData layerData)
        {
            _kernels = new Kernel[layerData.nKernels];
            for (int i = 0; i < _kernels.Length; i++)
            {
                _kernels[i] = new Kernel(inputDepth, layerData.kernelSize, layerData.activationFunctionType);
            }
        }

        Tensor IConvolutionalLayer.FeedForward(Tensor input)
        {
            Tensor output = new Tensor(_kernels.Length);

            for (int d = 0; d < _kernels.Length; d++)
            {
                IKernel kernel = _kernels[d];

                output[d] = kernel.Convolve(input);
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