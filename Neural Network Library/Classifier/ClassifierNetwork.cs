using Neural_Network_Library.Networks.ConvolutionalNeuralNetwork;
using Neural_Network_Library.Networks.MultilayeredPerceptron;

namespace Neural_Network_Library.Classifier
{
    public static class Classifier
    {
        public static ClassifierGuess Classify(this float[] input, INeuralNetwork neuralNetwork)
        {
            float[] output = neuralNetwork.FeedForward(input);
            output.NormalizeVector(output);

            return new ClassifierGuess(output);
        }

        public static ClassifierGuess Classify(this float[,] input, IConvolutionalNeuralNetwork neuralNetwork)
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
}