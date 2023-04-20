using static Neural_Network_Library.NeuralNetworkMath;

namespace Neural_Network_Library.MultilayeredPerceptron.NetworkTypes.Classifier
{
    public class ClassifierNetwork
    {
        private readonly NeuralNetwork _neuralNetwork;

        public ClassifierNetwork(NeuralNetwork neuralNetwork) => _neuralNetwork = neuralNetwork;

        public ClassifierGuess Classify(float[] input)
        {
            input = _neuralNetwork.FeedForward(input);
            NormalizeVector(input, input);

            return new ClassifierGuess(input);
        }
    }
}