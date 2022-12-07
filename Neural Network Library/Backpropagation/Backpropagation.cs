namespace Neural_Network_Library.Backpropagation
{
    public class Backpropagation
    {
        private readonly NeuralNetwork network;
        private readonly Datapoint[] dataset; // Manage dataset, where some is for training, some is for learning, automatically

        private readonly Layer[] networkLayers;
        private readonly BackpropagationLayer[] backpropagationLayers;
        private readonly BackpropagationOutputLayer outputLayer;

        private Cost c;

        public Backpropagation(NeuralNetwork network, Datapoint[] dataset)
        {
            this.dataset = dataset;
            this.network = network;
            networkLayers = network.layers;

            backpropagationLayers = new BackpropagationLayer[networkLayers.Length];

            outputLayer = new BackpropagationOutputLayer(networkLayers[^1]);
            backpropagationLayers[^1] = outputLayer;

            for (int i = backpropagationLayers.Length - 2; i >= 0; i--)
            {
                backpropagationLayers[i] = new BackpropagationHiddenLayer(networkLayers[i], backpropagationLayers[i + 1]);
            }

            c = new Cost(network);
        }

        public void Run(int iterations, int epochSize, float learnRate)
        {
            for (int i = 0; i < iterations; i++)
            {
                Train(i, epochSize, learnRate);
                c.PrintCost(dataset[36000..38000]); // Automatically manage this please!!
            }
        }

        private void Train(int i, int epochSize, float learnRate)
        {
            for (int j = 0; j < epochSize; j++)
            {
                int index = (i * epochSize + j) % dataset.Length;
                Backpropagate(dataset[index]);
            }

            for (int j = 0; j < backpropagationLayers.Length; j++)
            {
                backpropagationLayers[j].ApplyGradientVector(learnRate);
            }
        }

        private void Backpropagate(Datapoint datapoint)
        {
            network.FeedForward(datapoint.InputData);
            outputLayer.SetDesiredOutput(datapoint.DesiredOutput);

            for (int i = backpropagationLayers.Length - 1; i >= 0; i--)
            {
                backpropagationLayers[i].SumToGradientVector();
            }
        }
    }
}