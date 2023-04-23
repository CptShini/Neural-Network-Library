namespace Neural_Network_Library.Core.Math
{
    internal static class Single
    {
        internal static float Remap(this float value, float originLow, float originHigh, float destinationLow, float destinationHigh)
        {
            float originRange = originHigh - originLow;
            float destinationRange = destinationHigh - destinationLow;

            return (value - originLow) / originRange * destinationRange + destinationLow;
        }
    }
}
