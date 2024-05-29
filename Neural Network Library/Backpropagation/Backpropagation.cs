using Neural_Network_Library.Networks.MLP;

namespace Neural_Network_Library.Backpropagation
{
    public class Backpropagation
    {
        private readonly MultilayeredPerceptron _network;
        private readonly Datapoint[] _trainData;
        private readonly Datapoint[] _testData;

        private readonly MLPLayer[] _networkLayers;
        private readonly BackpropagationLayer[] _backpropagationLayers;
        private readonly BackpropagationOutputLayer _outputLayer;

        private readonly NetworkEvaluator _evaluator;

        public Backpropagation(MultilayeredPerceptron network, Datapoint[] trainData, Datapoint[] testData)
        {
            _trainData = trainData;
            _testData = testData;

            _network = network;
            _networkLayers = network._layers;

            _backpropagationLayers = new BackpropagationLayer[_networkLayers.Length];

            _outputLayer = new BackpropagationOutputLayer(_networkLayers[^1]);
            _backpropagationLayers[^1] = _outputLayer;

            for (int i = _backpropagationLayers.Length - 2; i >= 0; i--)
            {
                _backpropagationLayers[i] = new BackpropagationHiddenLayer(_networkLayers[i], _backpropagationLayers[i + 1]);
            }

            _evaluator = new NetworkEvaluator(network);
        }

        public void Run(int iterations, int epochSize, float learnRate, int evaluationPeriod)
        {
            for (int i = 0; i < iterations; i++)
            {
                Train(i, epochSize, learnRate);
                if (i % evaluationPeriod == 0)
                {
                    _evaluator.Evaulate(_testData);
                    _evaluator.PrintPerformance();
                }
            }
        }

        private void Train(int i, int epochSize, float learnRate)
        {
            for (int j = 0; j < epochSize; j++)
            {
                int index = (i * epochSize + j) % _trainData.Length;
                Backpropagate(_trainData[index]);
            }

            for (int j = 0; j < _backpropagationLayers.Length; j++)
            {
                _backpropagationLayers[j].ApplyGradientVector(learnRate);
            }
        }

        private void Backpropagate(Datapoint datapoint)
        {
            _network.FeedForward(datapoint.InputData);
            _outputLayer.SetDesiredOutput(datapoint.DesiredOutput);

            for (int i = _backpropagationLayers.Length - 1; i >= 0; i--)
            {
                _backpropagationLayers[i].SumToGradientVector();
            }
        }
    }
}