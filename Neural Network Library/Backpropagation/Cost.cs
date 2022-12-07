namespace Neural_Network_Library.Backpropagation
{
    public class Cost
    {
        private readonly NeuralNetwork network;

        public Cost(NeuralNetwork network) => this.network = network;

        public void PrintCost(Datapoint[] dataset)
        {
            float cost = GetNetworkCost(dataset);
            Console.WriteLine($"Cost: {cost}");
        }

        public float GetNetworkCost(Datapoint[] dataset)
        {
            float costSum = 0f;
            for (int i = 0; i < dataset.Length; i++)
            {
                //if (i < 10) GetDatapointCostPrint(dataset[i]);
                costSum += GetDatapointCost(dataset[i]);
            }

            return costSum / dataset.Length;
        }

        private float GetDatapointCost(Datapoint datapoint)
        {
            network.FeedForward(datapoint.InputData);

            float costSum = 0f;
            for (int j = 0; j < network.layers[^1].a.Length; j++)
            {
                float a = network.layers[^1].a[j];
                float y = datapoint.DesiredOutput[j];
                float cost = a - y;

                costSum += cost * cost;
            }

            return costSum;
        }

        private float GetDatapointCostPrint(Datapoint datapoint)
        {
            network.FeedForward(datapoint.InputData);

            float costSum = 0f;
            for (int j = 0; j < network.layers[^1].a.Length; j++)
            {
                float a = network.layers[^1].a[j];
                float y = datapoint.DesiredOutput[j];
                float cost = a - y;

                Console.WriteLine($"{a:0.0} - {y:0.0} = {cost:0.0} | {cost * cost:0.0} | {costSum}");

                costSum += cost * cost;
            }

            Console.WriteLine();

            return costSum;
        }
    }
}
