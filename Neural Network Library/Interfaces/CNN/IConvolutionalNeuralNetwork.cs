namespace Neural_Network_Library.Interfaces.CNN;

public interface IConvolutionalNeuralNetwork
{
    float[] FeedForward(float[,] input);
}