using static Neural_Network_Library.NeuralNetworkMath;

namespace Neural_Network_Library.MultilayeredPerceptron.NetworkTypes.Classifier
{
    public class ClassifierNetwork : NeuralNetwork
    {
        public ClassifierNetwork(int[] networkStructure, ActivationFunctionType activationFunction = ActivationFunctionType.Sigmoid) : base(networkStructure, activationFunction)
        {

        }

        public ClassifierGuess Classify(float[] input)
        {
            input = FeedForward(input);
            NormalizeVector(input);

            float confidence = input.Max();
            return new ClassifierGuess
            {
                GuessConfidence = confidence,
                GuessIndex = input.ToList().IndexOf(confidence),
                Outputs = input
            };
        }
    }
}