using Neural_Network_Library.ConvolutionalNeuralNetwork;

namespace Neural_Network_Library.Core.Math
{
    internal static class CNN
    {
        internal static Tensor Convolve(this Kernel[] kernels, Tensor input)
        {
            Tensor output = new Tensor(kernels.Length);

            for (int d = 0; d < output.Depth; d++)
            {
                output[d] = kernels[d].Convolve(input).MaxPool(2);
            }

            return output;
        }

        internal static float[,] Convolve(this Tensor input, Tensor filter)
        {
            int convolutionSize = input.Size - (filter.Size - 1);
            float[,] convolution = new float[convolutionSize, convolutionSize];

            for (int x = 0; x < convolutionSize; x++)
            {
                for (int y = 0; y < convolutionSize; y++)
                {
                    for (int i = 0; i < filter.Size; i++)
                    {
                        for (int j = 0; j < filter.Size; j++)
                        {
                            for (int d = 0; d < input.Depth; d++)
                            {
                                convolution[x, y] += input[d, x + i, y + j] * filter[d, i, j];
                            }
                        }
                    }
                }
            }

            return convolution;
        }

        internal static float[,] MaxPool(this float[,] input, int poolSize)
        {
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

        internal static float[] Flatten(this Tensor input)
        {
            int inputSize = input.Size;
            int outputLength = input.Volume;
            float[] output = new float[outputLength];

            int n = 0;
            for (int i = 0; i < input.Depth; i++)
            {
                for (int x = 0; x < inputSize; x++)
                {
                    for (int y = 0; y < inputSize; y++)
                    {
                        output[n++] = input[i, x, y];
                    }
                }
            }

            return output;
        }
    }
}
