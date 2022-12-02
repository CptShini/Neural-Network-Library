using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_Library
{
    public static class GeneticAlgorithm
    {

        public static NeuralNetwork[] GenerationLap(NeuralNetwork[] networks)
        {
            return GeneratePopulation(networks);
        }

        private static NeuralNetwork[] GeneratePopulation(NeuralNetwork[] population)
        {
            int n = population.Length / 2;

            NeuralNetwork[] children = CreateChildren(population, n);
            population = PurgePopulation(population, children);

            return population;
        }

        private static NeuralNetwork[] PurgePopulation(NeuralNetwork[] population, NeuralNetwork[] children)
        {
            population = population.OrderBy(n => n.Fitness).ToArray();

            for (int i = 0; i < children.Length; i++)
            {
                population[i] = children[i];
                population[i].Fitness = 0;
            }

            return population;
        }

        private static NeuralNetwork[] CreateChildren(NeuralNetwork[] nns, int n)
        {
            NeuralNetwork[] children = new NeuralNetwork[n];

            for (int i = 0; i < n; i++)
            {
                children[i] = CreateChild(nns);
            }

            return children;
        }

        private static NeuralNetwork CreateChild(NeuralNetwork[] nns)
        {
            NeuralNetwork[] parents = SelectParents(nns);
            NeuralNetwork child = new NeuralNetwork(parents[0], parents[1]);

            return child;
        }

        private static NeuralNetwork[] SelectParents(NeuralNetwork[] neuralNetworks)
        {
            NeuralNetwork[] parents = new NeuralNetwork[2];

            float totalFitness = GetTotalFitness(neuralNetworks);

            parents[0] = SelectNeuralNetwork(neuralNetworks, totalFitness);
            parents[1] = SelectNeuralNetwork(neuralNetworks, totalFitness);

            while (parents[0] == parents[1]) { parents[1] = SelectNeuralNetwork(neuralNetworks, totalFitness); }

            return parents;
        }

        private static NeuralNetwork SelectNeuralNetwork(NeuralNetwork[] neuralNetworks, float totalFitness)
        {
            float p = Rand(0f, totalFitness);

            float runningTotal = 0f;
            foreach (NeuralNetwork nn in neuralNetworks)
            {
                runningTotal += nn.Fitness;
                if (runningTotal > p) { return nn; }
            }

            throw new System.Exception("bruh");
        }

        private static float GetTotalFitness(NeuralNetwork[] neuralNetworks)
        {
            float totFitness = 0f;
            foreach (NeuralNetwork nn in neuralNetworks)
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
        }

    }
}
