namespace Neural_Network_Library.Core
{
    internal static class Math
    {
        internal static float Remap(this float value, float originLow, float originHigh, float destinationLow, float destinationHigh)
        {
            float originRange = originHigh - originLow;
            float destinationRange = destinationHigh - destinationLow;

            return (value - originLow) / originRange * destinationRange + destinationLow;
        }

        internal static float Max(this float[,] m)
        {
            float max = float.MinValue;

            for (int i = 0; i < m.GetLength(0); i++)
            {
                for (int j = 0; j < m.GetLength(1); j++)
                {
                    if (m[i, j] > max) max = m[i, j];
                }
            }

            return max;
        }

        internal static float Min(this float[,] m)
        {
            float min = float.MaxValue;

            for (int i = 0; i < m.GetLength(0); i++)
            {
                for (int j = 0; j < m.GetLength(1); j++)
                {
                    if (m[i, j] < min) min = m[i, j];
                }
            }

            return min;
        }
    }
}