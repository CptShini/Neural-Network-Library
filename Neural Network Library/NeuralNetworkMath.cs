namespace Neural_Network_Library
{
    internal static class NeuralNetworkMath
    {
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
    }
}
