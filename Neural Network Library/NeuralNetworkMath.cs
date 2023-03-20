namespace Neural_Network_Library
{
    internal static class NeuralNetworkMath
    {
        internal static float MatrixDotProduct(float[,] m1, float[,] m2)
        {
            float dotSum = 0f;

            for (int i = 0; i < m1.GetLongLength(0); i++)
            {
                for (int j = 0; j < m1.GetLongLength(1); j++)
                {
                    dotSum += m1[i, j] * m2[i, j];
                }
            }

            return dotSum;
        }

        internal static void MatrixVectorProduct(float[] outputVector, float[,] m, float[] v)
        {
            for (int j = 0; j < m.GetLongLength(0); j++)
            {
                outputVector[j] = 0f;

                for (int k = 0; k < m.GetLongLength(1); k++)
                {
                    outputVector[j] += m[j, k] * v[k];
                }
            }
        }

        internal static void AddVectors(float[] outputVector, float[] v1, float[] v2)
        {
            for (int i = 0; i < v1.Length; i++)
            {
                outputVector[i] = v1[i] + v2[i];
            }
        }

        internal static void SumVectors(float[] outputVector, params float[][] vectors)
        {
            for (int i = 0; i < vectors.Length; i++)
            {
                outputVector[i] = 0f;

                for (int j = 0; j < vectors[i].Length; j++)
                {
                    outputVector[i] += vectors[i][j];
                }
            }
        }

        internal static void NormalizeVector(float[] outputVector)
        {
            float outputSum = outputVector.Sum();
            for (int i = 0; i < outputVector.Length; i++)
            {
                outputVector[i] /= outputSum;
            }
        }

        internal static float Convolve(float[,] input, float[,] filter, int x1, int x2, int y1, int y2)
        {
            float convolutionSum = 0f;

            for (int i = x1; i < x2; i++)
            {
                for (int j = y1; j < y2; j++)
                {
                    convolutionSum += inputMatrix[i, j];
                }
            }

            return convolutionSum;
        }

        internal static float MaxPool(float[,] inputMatrix, int x1, int x2, int y1, int y2)
        {
            float currentMax = 0f;

            for (int i = x1; i < x2; i++)
            {
                for (int j = y1; j < y2; j++)
                {
                    if (inputMatrix[i, j] > currentMax) currentMax = inputMatrix[i, j];
                }
            }

            return currentMax;
        }
    }
}
