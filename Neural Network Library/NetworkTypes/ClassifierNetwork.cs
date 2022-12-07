namespace Neural_Network_Library.NetworkTypes
{
    public class ClassifierNetwork : NeuralNetwork
    {
        public int Guess;
        public float Confidence;

        public ClassifierNetwork(int[] networkStructure, ActivationFunctionType activationFunction = ActivationFunctionType.Sigmoid) : base(networkStructure, activationFunction)
        {

        }

        public float[] Classify(float[] input)
        {
            input = FeedForward(input);

            float outputSum = input.Sum();
            for (int i = 0; i < input.Length; i++)
            {
                input[i] /= outputSum;
            }

            Confidence = input.Max();
            Guess = input.ToList().IndexOf(Confidence);

            return input;
        }
    }
}