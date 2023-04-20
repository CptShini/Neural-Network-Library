namespace Neural_Network_Library.MultilayeredPerceptron.NetworkTypes.Classifier
{
    public struct ClassifierGuess
    {
        public readonly int GuessIndex;
        public readonly float GuessConfidence;
        public readonly float[] Outputs;

        public ClassifierGuess(float[] networkOutput)
        {
            GuessConfidence = networkOutput.Max();
            GuessIndex = networkOutput.ToList().IndexOf(GuessConfidence);
            Outputs = networkOutput;
        }
    }
}
