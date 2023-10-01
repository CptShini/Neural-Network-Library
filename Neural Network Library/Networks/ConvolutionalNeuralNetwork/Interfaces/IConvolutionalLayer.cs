namespace Neural_Network_Library.Networks.ConvolutionalNeuralNetwork
{
    internal interface IConvolutionalLayer
    {
        Tensor FeedForward(Tensor input);
    }
}