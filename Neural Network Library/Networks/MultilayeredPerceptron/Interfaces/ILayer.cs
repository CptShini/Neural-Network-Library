namespace Neural_Network_Library.Networks.MultilayeredPerceptron
{
    internal interface ILayer
    {
        float[] FeedForward(float[] input);
    }
}