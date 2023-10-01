namespace Neural_Network_Library.SaveLoad
{
    public abstract class NetworkSaveLoader<T>
    {
        #region Saving

        public void SaveNetwork(string name, string path)
        {
            string encodedNetwork = EncodeNetwork();
            File.WriteAllText(@$"{path}\{name}.txt", encodedNetwork);
        }

        private protected abstract string EncodeNetwork();

        #endregion

        #region Loading

        /*public T LoadNetwork(string path)
        {
            string encodedNetwork = File.ReadAllText(path);

            T decodedNetwork = DecodeNetwork(encodedNetwork);

            return decodedNetwork;
        }

        private protected abstract T DecodeNetwork(string encodedNetwork);*/

        #endregion
    }
}