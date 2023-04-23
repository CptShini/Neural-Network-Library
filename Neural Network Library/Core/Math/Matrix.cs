namespace Neural_Network_Library.Core.Math
{
    internal static class Matrix
    {
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
    }
}
