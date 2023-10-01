using Neural_Network_Library.Networks.MultilayeredPerceptron;

namespace Neural_Network_Library.Backpropagation
{
    internal class NetworkEvaluator
    {
        private readonly NeuralNetwork _network;
        private float _precision;
        private float _cost;

        internal NetworkEvaluator(NeuralNetwork network) => _network = network;

        internal float GetPrecision() => _precision;

        internal float GetCost() => _cost;

        internal void PrintCost() => Console.WriteLine($"Cost: {_cost}");

        internal void PrintPrecision() => Console.WriteLine($"Precision: {_precision * 100:00.00}%");

        internal void PrintPerformance() => Console.WriteLine($"Performance: {_precision * 100:00.00}% | {_cost}");

        internal void Evaulate(Datapoint[] dataset)
        {
            float costSum = 0f;
            float precisionSum = 0;
            foreach (Datapoint datapoint in dataset)
            {
                float[] desiredOutput = datapoint.DesiredOutput;
                float[] output = _network.FeedForward(datapoint.InputData);

                costSum += GetDatapointCost(desiredOutput, output);
                precisionSum += GetDatapointPrecision(desiredOutput, output);
            }

            _cost = costSum / dataset.Length;
            _precision = precisionSum / dataset.Length;
        }

        private static float GetDatapointPrecision(float[] dersiredOutput, float[] output)
        {
            int guess = output.ToList().IndexOf(output.Max());
            return dersiredOutput[guess];
        }

        private static float GetDatapointCost(float[] dersiredOutput, float[] output)
        {
            float costSum = 0f;
            for (int j = 0; j < output.Length; j++)
            {
                float a = output[j];
                float y = dersiredOutput[j];
                float cost = a - y;

                costSum += cost * cost;
            }

            return costSum;
        }
    }
}
