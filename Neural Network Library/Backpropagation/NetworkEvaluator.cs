namespace Neural_Network_Library.Backpropagation
{
    internal class NetworkEvaluator
    {
        private readonly NeuralNetwork network;
        private float precision;
        private float cost;

        internal NetworkEvaluator(NeuralNetwork network) => this.network = network;

        internal void PrintCost() => Console.WriteLine($"Cost: {cost}");

        internal void PrintPrecision() => Console.WriteLine($"Precision: {precision * 100:00.00}%");

        internal void PrintPerformance() => Console.WriteLine($"Performance: {precision * 100:00.00}% | {cost}");

        internal void Evaulate(Datapoint[] dataset)
        {
            float costSum = 0f;
            float precisionSum = 0;
            foreach (Datapoint datapoint in dataset)
            {
                float[] desiredOutput = datapoint.DesiredOutput;
                float[] output = network.FeedForward(datapoint.InputData);

                costSum += GetDatapointCost(desiredOutput, output);
                precisionSum += GetDatapointPrecision(desiredOutput, output);
            }

            cost = costSum / dataset.Length;
            precision = precisionSum / dataset.Length;
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
