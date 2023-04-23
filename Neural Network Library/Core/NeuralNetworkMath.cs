using Neural_Network_Library.ConvolutionalNeuralNetwork;

namespace Neural_Network_Library.Core
{
    internal static class NeuralNetworkMath
    {
        internal static float MatrixDotProduct(float[,] m1, float[,] m2)
        {
            float dotSum = 0f;

            for (int i = 0; i < m1.GetLength(0); i++)
            {
                for (int j = 0; j < m1.GetLength(1); j++)
                {
                    dotSum += m1[i, j] * m2[i, j];
                }
            }

            return dotSum;
        }

        internal static void MatrixVectorProduct(this float[] outputVector, float[,] m, float[] v)
        {
            for (int j = 0; j < m.GetLength(0); j++)
            {
                outputVector[j] = 0f;

                for (int k = 0; k < m.GetLength(1); k++)
                {
                    outputVector[j] += m[j, k] * v[k];
                }
            }
        }

        internal static void AddVector(this float[] v1, float[] v2)
        {
            for (int i = 0; i < v1.Length; i++)
            {
                v1[i] += v2[i];
            }
        }

        internal static void SumVectors(float[] outputVector, params float[][] vectors)
        {
            for (int i = 0; i < vectors[0].Length; i++)
            {
                outputVector[i] = 0f;
                for (int n = 0; n < vectors.Length; n++)
                {
                    outputVector[i] += vectors[n][i];
                }
            }
        }

        internal static void AddMatrix(this float[,] m1, float[,] m2)
        {
            for (int i = 0; i < m1.GetLength(0); i++)
            {
                for (int j = 0; j < m1.GetLength(1); j++)
                {
                    m1[i, j] += m2[i, j];
                }
            }
        }

        internal static void AddToMatrix(this float[,] m1, float val)
        {
            for (int i = 0; i < m1.GetLength(0); i++)
            {
                for (int j = 0; j < m1.GetLength(1); j++)
                {
                    m1[i, j] += val;
                }
            }
        }

        internal static void SumMatricies(float[,] outputMatrix, params float[][,] matricies)
        {
            for (int i = 0; i < outputMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < outputMatrix.GetLength(1); j++)
                {
                    outputMatrix[i, j] = 0f;
                    for (int n = 0; n < matricies.Length; n++)
                    {
                        outputMatrix[i, j] += matricies[n][i, j];
                    }
                }
            }
        }

        internal static void NormalizeVector(this float[] outputVector, float[] inputVector)
        {
            float inputSum = inputVector.Sum();
            for (int i = 0; i < inputVector.Length; i++)
            {
                outputVector[i] = inputVector[i] / inputSum;
            }
        }

        internal static float Remap(this float value, float originLow, float originHigh, float destinationLow, float destinationHigh)
        {
            float originRange = originHigh - originLow;
            float destinationRange = destinationHigh - destinationLow;

            return (value - originLow) / originRange * destinationRange + destinationLow;
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
    }
}