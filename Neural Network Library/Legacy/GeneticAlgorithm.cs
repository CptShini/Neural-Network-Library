namespace Neural_Network_Library.Legacy
{
    public static class GeneticAlgorithm
    {

        /*public static NeuralNetworkLib[] GenerationLap(NeuralNetworkLib[] networks)
        {
            return GeneratePopulation(networks);
        }

        private static NeuralNetworkLib[] GeneratePopulation(NeuralNetworkLib[] population)
        {
            int n = population.Length / 2;

            NeuralNetworkLib[] children = CreateChildren(population, n);
            population = PurgePopulation(population, children);

            return population;
        }

        private static NeuralNetworkLib[] PurgePopulation(NeuralNetworkLib[] population, NeuralNetworkLib[] children)
        {
            population = population.OrderBy(n => n.Fitness).ToArray();

            for (int i = 0; i < children.Length; i++)
            {
                population[i] = children[i];
                population[i].Fitness = 0;
            }

            return population;
        }

        private static NeuralNetworkLib[] CreateChildren(NeuralNetworkLib[] nns, int n)
        {
            NeuralNetworkLib[] children = new NeuralNetworkLib[n];

            for (int i = 0; i < n; i++)
            {
                children[i] = CreateChild(nns);
            }

            return children;
        }

        private static NeuralNetworkLib CreateChild(NeuralNetworkLib[] nns)
        {
            NeuralNetworkLib[] parents = SelectParents(nns);
            NeuralNetworkLib child = new NeuralNetworkLib(parents[0], parents[1]);

            return child;
        }

        private static NeuralNetworkLib[] SelectParents(NeuralNetworkLib[] neuralNetworks)
        {
            NeuralNetworkLib[] parents = new NeuralNetworkLib[2];

            float totalFitness = GetTotalFitness(neuralNetworks);

            parents[0] = SelectNeuralNetwork(neuralNetworks, totalFitness);
            parents[1] = SelectNeuralNetwork(neuralNetworks, totalFitness);

            while (parents[0] == parents[1]) { parents[1] = SelectNeuralNetwork(neuralNetworks, totalFitness); }

            return parents;
        }

        private static NeuralNetworkLib SelectNeuralNetwork(NeuralNetworkLib[] neuralNetworks, float totalFitness)
        {
            float p = Rand(0f, totalFitness);

            float runningTotal = 0f;
            foreach (NeuralNetworkLib nn in neuralNetworks)
            {
                runningTotal += nn.Fitness;
                if (runningTotal > p) { return nn; }
            }

            throw new Exception("bruh");
        }

        private static float GetTotalFitness(NeuralNetworkLib[] neuralNetworks)
        {
            float totFitness = 0f;
            foreach (NeuralNetworkLib nn in neuralNetworks)
            {
                totFitness += nn.Fitness;
            }
            return totFitness;
        }

        public static float Rand(float min, float max)
        {
            Random r = new Random();
            float val = (float)r.NextDouble();
            return Remap(val, 0, 1, min, max);
        }

        public static int Rand(int min, int max)
        {
            Random r = new Random();
            int val = r.Next(min, max);
            return val;
        }

        public static float Remap(float value, float from1, float to1, float from2, float to2)
        {
            return (value - from1) / (to1 - from1) * (to2 - from2) + from2;
        }*/

    }
}
