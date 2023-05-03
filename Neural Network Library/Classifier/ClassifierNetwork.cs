using Neural_Network_Library.Core;

namespace Neural_Network_Library.Classifier
{
    public static class Classifier
    {
        public static ClassifierGuess Classify(this float[] input, MultilayeredPerceptron.NeuralNetwork neuralNetwork)
        {
            float[] output = neuralNetwork.FeedForward(input);
            output.NormalizeVector(output);

            return new ClassifierGuess(output);
        }

        public static ClassifierGuess Classify(this float[,] input, ConvolutionalNeuralNetwork.ConvolutionalNeuralNetwork neuralNetwork)
        {
            float[] output = neuralNetwork.FeedForwardTest(input);
            output.NormalizeVector(output);

            return new ClassifierGuess(output);
        }
    }
}