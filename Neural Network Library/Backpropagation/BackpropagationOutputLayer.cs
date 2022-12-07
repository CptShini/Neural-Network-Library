namespace Neural_Network_Library.Backpropagation
{
    internal class BackpropagationOutputLayer : BackpropagationLayer
    {
        private float[] y;

        public BackpropagationOutputLayer(Layer layer) : base(layer) => y = Array.Empty<float>();

        internal void SetDesiredOutput(float[] desiredOutput) => y = desiredOutput;

        protected override float Calculate_da(int k) => 2 * (Layer.a[k] - y[k]);
    }
}
