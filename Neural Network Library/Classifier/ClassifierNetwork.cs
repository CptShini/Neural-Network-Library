using static Neural_Network_Library.Core.NeuralNetworkMath;

namespace Neural_Network_Library.Classifier
{
    public static class Classifier
    {
        public static ClassifierGuess Classify(float[] input, MultilayeredPerceptron.NeuralNetwork neuralNetwork)
        {
            float[] output = neuralNetwork.FeedForward(input);
            NormalizeVector(output, output);

            return new ClassifierGuess(output);
        }

        public static ClassifierGuess Classify(float[,] input, ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork neuralNetwork)
        {
            float[] output = neuralNetwork.FeedForward(input);
            NormalizeVector(output, output);

            return new ClassifierGuess(output);
        }
    }
}