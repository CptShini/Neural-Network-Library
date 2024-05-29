using Neural_Network_Library.Networks.MLP;

namespace Neural_Network_Library.Backpropagation
{
    internal class BackpropagationOutputLayer : BackpropagationLayer
    {
        private float[] _y;

        internal BackpropagationOutputLayer(MLPLayer layer) : base(layer) => _y = Array.Empty<float>();

        internal void SetDesiredOutput(float[] desiredOutput) => _y = desiredOutput;

        private protected override float Calculate_da(int k) => 2 * (_layer._a[k] - _y[k]);
    }
}
