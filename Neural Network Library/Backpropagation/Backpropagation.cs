namespace Neural_Network_Library.Backpropagation
{
    public class Backpropagation
    {
        private readonly NeuralNetwork network;
        private readonly Datapoint[] trainData;
        private readonly Datapoint[] testData;

        private readonly Layer[] networkLayers;
        private readonly BackpropagationLayer[] backpropagationLayers;
        private readonly BackpropagationOutputLayer outputLayer;

        private readonly NetworkEvaluator evaluator;

        public Backpropagation(NeuralNetwork network, Datapoint[] trainData, Datapoint[] testData)
        {
            this.trainData = trainData;
            this.testData = testData;

            this.network = network;
            networkLayers = network.layers;

            backpropagationLayers = new BackpropagationLayer[networkLayers.Length];

            outputLayer = new BackpropagationOutputLayer(networkLayers[^1]);
            backpropagationLayers[^1] = outputLayer;

            for (int i = backpropagationLayers.Length - 2; i >= 0; i--)
            {
                backpropagationLayers[i] = new BackpropagationHiddenLayer(networkLayers[i], backpropagationLayers[i + 1]);
            }

            evaluator = new NetworkEvaluator(network);
        }

        public void Run(int iterations, int epochSize, float learnRate, int evaluationPeriod)
        {
            for (int i = 0; i < iterations; i++)
            {
                Train(i, epochSize, learnRate);
                if (i % evaluationPeriod == 0)
                {
                    evaluator.Evaulate(testData);
                    evaluator.PrintPerformance();
                }
            }
        }

        private void Train(int i, int epochSize, float learnRate)
        {
            for (int j = 0; j < epochSize; j++)
            {
                int index = (i * epochSize + j) % trainData.Length;
                Backpropagate(trainData[index]);
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