namespace Neural_Network_Library.Core.Math
{
    internal static class Vector
    {
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

        internal static void NormalizeVector(this float[] outputVector, float[] inputVector)
        {
            float inputSum = inputVector.Sum();
            for (int i = 0; i < inputVector.Length; i++)
            {
                outputVector[i] = inputVector[i] / inputSum;
            }
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
    }
}
