namespace Neural_Network_Library
{
    public static class Random
    {
        private static readonly System.Random r = new();

        public static float Range(float min, float max) => (float)r.NextDouble() * (max - min) + min;

        public static int Range(int min, int max) => r.Next(min, max);

        public static double Range(double min, double max) => r.NextDouble() * (max - min) + min;
    }
}