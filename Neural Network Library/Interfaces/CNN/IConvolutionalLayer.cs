using Neural_Network_Library.Networks.CNN;

namespace Neural_Network_Library.Interfaces.CNN;

internal interface IConvolutionalLayer
{
    Tensor FeedForward(Tensor input);
}