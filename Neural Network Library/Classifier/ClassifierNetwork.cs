using Neural_Network_Library.Networks.CNN;
using Neural_Network_Library.Networks.MLP;

namespace Neural_Network_Library.Classifier;

public static class Classifier
{
    public static ClassifierGuess Classify(this float[] input, MultilayeredPerceptron neuralNetwork)
    {
        float[] output = neuralNetwork.FeedForward(input);
        output.NormalizeVector(output);

        return new ClassifierGuess(output);
    }

    public static ClassifierGuess Classify(this float[,] input, ConvolutionalNeuralNetwork neuralNetwork)
    {
        float[] output = neuralNetwork.FeedForward(input);
        output.NormalizeVector(output);

        return new ClassifierGuess(output);
    }

    private static void NormalizeVector(this float[] outputVector, float[] inputVector)
    {
        float inputSum = inputVector.Sum();
        for (int i = 0; i < inputVector.Length; i++)
        {
            outputVector[i] = inputVector[i] / inputSum;
        }
    }
}