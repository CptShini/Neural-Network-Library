using Neural_Network_Library.Networks.MultilayeredPerceptron;

namespace Neural_Network_Library.Backpropagation
{
    internal class BackpropagationHiddenLayer : BackpropagationLayer
    {
        private readonly BackpropagationLayer _prevLayer;

        internal BackpropagationHiddenLayer(Layer layer, BackpropagationLayer prevLayer) : base(layer) => _prevLayer = prevLayer;

        private protected override float Calculate_da(int k) => _prevLayer.Calculate_NextLayer_da(k);
    }
}