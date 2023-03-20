namespace Neural_Network_Library
{
    public static class Random
    {
        private static readonly System.Random _r = new();

        public static float Range(float min, float max) => (float)_r.NextDouble() * (max - min) + min;

        public static float Range(float max) => (float)_r.NextDouble() * max;

        public static int Range(int min, int max) => _r.Next(min, max);

        public static int Range(int max) => _r.Next(max);

        public static double Range(double min, double max) => _r.NextDouble() * (max - min) + min;

        public static double Range(double max) => _r.NextDouble() * max;
    }
}