namespace Neural_Network_Library.Backpropagation
{
    internal class BackpropagationHiddenLayer : BackpropagationLayer
    {
        private readonly BackpropagationLayer prevLayer;

        internal BackpropagationHiddenLayer(Layer layer, BackpropagationLayer prevLayer) : base(layer) => this.prevLayer = prevLayer;

        protected override float Calculate_da(int k) => prevLayer.Calculate_NextLayer_da(k);
    }
}
