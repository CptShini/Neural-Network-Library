namespace Neural_Network_Library.Backpropagation
{
    internal abstract class BackpropagationLayer
    {
        protected readonly Layer Layer;

        protected readonly float[] Da, Db;
        protected readonly float[,] Dw;

        private int epochCount;

        protected BackpropagationLayer(Layer layer)
        {
            Layer = layer;

            Da = new float[layer.a.Length];
            Db = new float[layer.b.Length];
            Dw = new float[layer.w.GetLength(0), layer.w.GetLength(1)];

            epochCount = 0;
        }

        internal void SumToGradientVector()
        {
            for (int j = 0; j < Da.Length; j++)
            {
                Da[j] += Calculate_da(j);

                float dz = Layer.DerivedActivate(Layer.z[j]) * Da[j];
                Db[j] += dz;

                for (int k = 0; k < Dw.GetLength(1); k++)
                {
                    Dw[j, k] += Layer.a_1[k] * dz;
                }
            }

            epochCount++;
        }

        internal void ApplyGradientVector(float learnRate)
        {
            for (int j = 0; j < Layer.b.Length; j++)
            {
                for (int k = 0; k < Layer.w.GetLength(1); k++)
                {
                    Layer.w[j, k] -= Dw[j, k] / epochCount * learnRate;
                    Dw[j, k] = 0f;
                }

                Layer.b[j] -= Db[j] / epochCount * learnRate;
                Db[j] = 0f;

                Da[j] = 0f;
            }
            
            epochCount = 0;
        }

        protected abstract float Calculate_da(int k);

        internal float Calculate_NextLayer_da(int k)
        {
            float sum = 0f;
            for (int j = 0; j < Da.Length; j++)
            {
                sum += Layer.w[j, k] * Layer.DerivedActivate(Layer.z[j]) * Da[j];
            }

            return sum;
        }
    }
}