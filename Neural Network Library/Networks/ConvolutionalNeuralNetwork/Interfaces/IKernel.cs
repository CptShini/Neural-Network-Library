namespace Neural_Network_Library.Networks.ConvolutionalNeuralNetwork
{
    internal interface IKernel
    {
        float[,] Convolve(Tensor input);
    }
}