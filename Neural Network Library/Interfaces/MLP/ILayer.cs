namespace Neural_Network_Library.Interfaces.MLP;

internal interface ILayer
{
    float[] FeedForward(float[] input);
}