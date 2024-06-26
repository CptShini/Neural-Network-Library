﻿using Neural_Network_Library.Classifier;
using Neural_Network_Library.Networks.CNN;
using Neural_Network_Library.Core;
using Neural_Network_Library.Networks.MLP;
using Neural_Network_Library.Backpropagation;
using static Neural_Network_Library.Core.RandomNumberGenerator;
using Neural_Network_Library.SaveLoad;
using System.IO;

namespace NumImgTest
{
    public class Program
    {
        private static void Main(string[] args)
        {
            TestBackpropagation();
            //TestConvolutional();
        }

        static void TestConvolutional()
        {
            CNNStructure structure = new CNNStructure(28, 10);
            structure.AddLayer(2, 5, ActivationFunctionType.ReLU);
            structure.AddLayer(4, 3, ActivationFunctionType.Sigmoid);

            ConvolutionalNeuralNetwork CNN = new ConvolutionalNeuralNetwork(structure);
            
            Datapoint[] dataset = ImportDataset(@"C:\Users\gabri\Documents\Code Shit\train.csv");
            for (int i = 0; i < 4; i++)
            {
                float[,] input = ParseInputData(dataset[RandomRange(10000)].InputData);
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
            Datapoint[] dataset = ImportDataset(@"C:\Users\gabri\Documents\Code Shit\train.csv");
            Datapoint[] trainset = dataset[0..40000];
            Datapoint[] testset = dataset[40000..42000];

            MLPStructure structure = new MLPStructure(784);
            structure.AddLayer(16);
            structure.AddLayer(16);
            structure.AddLayer(10);

            MultilayeredPerceptron neuralNetwork = new MultilayeredPerceptron(structure);
            Backpropagation backpropagation = new Backpropagation(neuralNetwork, trainset, testset);

            MultilayeredPerceptronSaveLoader mlpS = new MultilayeredPerceptronSaveLoader(neuralNetwork);

            mlpS.SaveNetwork("Network before training", @"C:\Users\gabri\Documents\Code Shit\TestFolder");
            backpropagation.Run(10000, 100, 0.5f, 200);
            mlpS.SaveNetwork("Network after training", @"C:\Users\gabri\Documents\Code Shit\TestFolder");

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
