using Neural_Network_Library.Networks.ConvolutionalNeuralNetwork;

namespace Neural_Network_Library.SaveLoad
{
    public class ConvolutionalNeuralNetworkSaveLoader : NetworkSaveLoader<ConvolutionalNeuralNetwork>
    {
        private readonly ConvolutionalNeuralNetwork _network;

        public ConvolutionalNeuralNetworkSaveLoader(ConvolutionalNeuralNetwork network) => _network = network;

        private protected override string EncodeNetwork() => throw new NotImplementedException();
    }
}