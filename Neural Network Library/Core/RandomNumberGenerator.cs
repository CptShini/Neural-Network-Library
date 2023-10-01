namespace Neural_Network_Library.Core
{
    public static class RandomNumberGenerator
    {
        private static readonly Random _r = new();

        public static float RandomRange(float min, float max) => (float)_r.NextDouble() * (max - min) + min;

        public static float RandomRange(float max) => (float)_r.NextDouble() * max;

        public static int RandomRange(int min, int max) => _r.Next(min, max);

        public static int RandomRange(int max) => _r.Next(max);

        public static double RandomRange(double min, double max) => _r.NextDouble() * (max - min) + min;

        public static double RandomRange(double max) => _r.NextDouble() * max;
    }
}