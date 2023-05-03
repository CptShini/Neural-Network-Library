using Neural_Network_Library.Classifier;
using Neural_Network_Library.ConvolutionalNeuralNetwork;
using Neural_Network_Library.Core;
using Neural_Network_Library.MultilayeredPerceptron;
using Neural_Network_Library.MultilayeredPerceptron.Backpropagation;
using Random = Neural_Network_Library.Core.Random;

namespace NumImgTest
{
    public class Program
    {
        private static void Main(string[] args)
        {
            //TestBackpropagation();
            TestConvolutional();
        }

        static void TestConvolutional()
        {
            CNNStructure structure = new CNNStructure();
            structure.AddLayer(2, 5, ActivationFunctionType.ReLU);
            structure.AddLayer(4, 3, ActivationFunctionType.Sigmoid);

            ConvolutionalNeuralNetwork CNN = new ConvolutionalNeuralNetwork(28, 10, structure);

            Datapoint[] dataset = ImportDataset(@"C:\Users\gabri\Desktop\Code Shit\train.csv");
            for (int i = 0; i < 1; i++)
            {
                float[,] input = ParseInputData(dataset[Random.Range(10000)].InputData);
                ClassifierGuess output = input.Classify(CNN);
                for (int j = 0; j < output.Count; j++)
                {
                    Console.WriteLine($"{output[j].Key} | {output[j].Value * 100:00.00}%");
                }
                Console.WriteLine();
            }
        }

        static float[,] ParseInputData(float[] inputData)
        {
            float[,] input = new float[28, 28];
            for (int i = 0; i < inputData.Length; i++)
            {
                input[i / 28, i % 28] = inputData[i];
            }

            return input;
        }

        static void TestBackpropagation()
        {
            Datapoint[] dataset = ImportDataset(@"C:\Users\gabri\Desktop\Code Shit\train.csv");
            Datapoint[] trainset = dataset[0..40000];
            Datapoint[] testset = dataset[40000..42000];

            int[] layers = { 784, 16, 16, 10 };
            NeuralNetwork neuralNetwork = new NeuralNetwork(layers);
            Backpropagation backpropagation = new Backpropagation(neuralNetwork, trainset, testset);

            backpropagation.Run(10000, 100, 0.5f, 200);

            foreach (Datapoint datapoint in testset)
            {
                ClassifierGuess output = datapoint.InputData.Classify(neuralNetwork);

                int answer = datapoint.DesiredOutput.ToList().IndexOf(datapoint.DesiredOutput.Max());
                for (int j = 0; j < output.Count; j++)
                {
                    Console.WriteLine($"{answer} | {output[j].Key} | {output[j].Value * 100:00.00}%");
                }
                Console.WriteLine();
            }
        }

        static Datapoint[] ImportDataset(string path)
        {
            string[] datasetFile = File.ReadAllLines(path);
            Datapoint[] dataset = new Datapoint[datasetFile.Length - 1];

            for (int i = 0; i < dataset.Length; i++)
            {
                string datapoint = datasetFile[i + 1];
                string[] datapoints = datapoint.Split(",");

                float[] answer = new float[10];
                answer[int.Parse(datapoints[0])] = 1f;

                float[] data = new float[datapoints.Length - 1];
                for (int j = 0; j < datapoints.Length - 1; j++)
                {
                    data[j] = float.Parse(datapoints[j + 1]) / 255f;
                }

                dataset[i] = new Datapoint(data, answer);
            }

            return dataset;
        }
    }
}
