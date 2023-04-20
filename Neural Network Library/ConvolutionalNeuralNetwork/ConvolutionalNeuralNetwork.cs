namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    public class ConvolutionalNeuralNetwork
    {
        private readonly ConvolutionalLayer[] _layers;

        public ConvolutionalNeuralNetwork(CNNStructure networkStructure) => _layers = networkStructure.GetLayerStructure();

        public float[][,] GetOutputs(float[,] input)
        {
            float[][,] output = new float[][,] { input };

            foreach (ConvolutionalLayer layer in _layers)
            {
                output = layer.Process(output);
            }

            return output;
        }
    }
}
