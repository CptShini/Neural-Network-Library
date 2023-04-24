namespace Neural_Network_Library.Classifier
{
    public struct ClassifierGuess
    {
        private readonly Dictionary<int, float> _networkGuesses;

        public int Count => _networkGuesses.Count;

        internal ClassifierGuess(float[] networkOutput)
        {
            _networkGuesses = new Dictionary<int, float>();

            for (int i = 0; i < networkOutput.Length; i++)
            {
                _networkGuesses.Add(i, networkOutput[i]);
            }

            _networkGuesses = _networkGuesses.OrderByDescending(g => g.Value).ToDictionary(g => g.Key, g => g.Value);
        }

        public KeyValuePair<int, float> this[int i] => _networkGuesses.ElementAt(i);
    }
}
