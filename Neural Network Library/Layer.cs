namespace Neural_Network_Library
{
    internal class Layer
    {
        internal readonly float[] a, z, b;
        internal readonly float[,] w;
        internal float[] a_1;

        private readonly ActivationFunctionType activationFunctionType;

        internal Layer(int inputSize, int outputSize, ActivationFunctionType activationFunctionType = ActivationFunctionType.Sigmoid)
        {
            a = new float[outputSize];
            z = new float[outputSize];
            b = new float[outputSize];
            w = new float[outputSize, inputSize];
            a_1 = new float[outputSize];

            this.activationFunctionType = activationFunctionType;

            InitializeWeightsAndBiases();
        }

        private void InitializeWeightsAndBiases()
        {
            for (int j = 0; j < w.GetLength(0); j++)
            {
                for (int k = 0; k < w.GetLength(1); k++)
                {
                    w[j, k] = Random.Range(-1f, 1f);
                }

                b[j] = Random.Range(-1f, 1f);
            }
        }

        internal float[] FeedForward(float[] input)
        {
            a_1 = input;

            for (int j = 0; j < z.Length; j++)
            {
                z[j] = b[j];
                for (int k = 0; k < a_1.Length; k++)
                {
                    z[j] += w[j, k] * a_1[k];
                }

                a[j] = Activate(z[j]);
            }

            return a;
        }

        internal float Activate(float val) => ActivationFunction.Activate(val, activationFunctionType);
        internal float DerivedActivate(float val) => ActivationFunction.DerivedActive(val, activationFunctionType);
    }
}