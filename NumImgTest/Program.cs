using Neural_Network_Library.Classifier;
using Neural_Network_Library.ConvolutionalNeuralNetwork;
using Neural_Network_Library.Core;
using Neural_Network_Library.MultilayeredPerceptron;
using Neural_Network_Library.MultilayeredPerceptron.Backpropagation;
using System.Drawing;
using Random = Neural_Network_Library.Core.Random;

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
            float[] dataset = ImportDataset(@"C:\Users\gabri\Desktop\Code Shit\train.csv")[Random.Range(100)].InputData;

            float[,] input = new float[28, 28];
            for (int i = 0; i < dataset.Length; i++)
            {
                input[i / 28, i % 28] = dataset[i];
            }

            CNNStructure CNNStructure = new CNNStructure();
            CNNStructure.AddLayer(2, 5, ActivationFunctionType.ReLU);
            CNNStructure.AddLayer(4, 3, ActivationFunctionType.Sigmoid);

            ConvolutionalNeuralNetwork CNN = new ConvolutionalNeuralNetwork(28, 10, CNNStructure);

            ClassifierGuess output = Classifier.Classify(input, CNN);
            Console.WriteLine($"{output.GuessIndex} | {output.GuessConfidence * 100:00.00}%");
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
                ClassifierGuess guess = Classifier.Classify(datapoint.InputData, neuralNetwork);

                int answer = datapoint.DesiredOutput.ToList().IndexOf(datapoint.DesiredOutput.Max());
                Console.WriteLine($"{answer} | {guess.GuessIndex} | {guess.GuessConfidence * 100:00}%");
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
        
        static void SaveFloatArrayAsBitmap(float[] pixels)
        {
            Bitmap bmp = new Bitmap(28, 28);
            
            for (int i = 0; i < pixels.Length; i++)
            {
                int val = (int)(pixels[i] * 255f);

                int x = i % 28;
                int y = i / 28;
                Color color = Color.FromArgb(val, val, val);
                
                bmp.SetPixel(x, y, color);
            }
            bmp.Save($@"C:\Users\gabri\Desktop\Code Shit\TestFolder\{DateTime.Now.Ticks}.png");
        }

        static void SaveFloatMatrixAsBitmap(float[,] pixels, string name)
        {
            int scaler = 10;

            Bitmap bmp = new Bitmap(28 * scaler, 28 * scaler);
            for (int i = 0; i < pixels.GetLongLength(0); i++)
            {
                for (int j = 0; j < pixels.GetLongLength(1); j++)
                {
                    float val = pixels[j, i];
                    int intensity = (int)(val * 255);
                    intensity = (int)MathF.Min(intensity, 255);

                    Color color = (val <= 255) ? Color.FromArgb(intensity, intensity, intensity) : Color.FromArgb(255, 0, 0);
                    for (int iS = 0; iS < scaler; iS++)
                    {
                        for (int jS = 0; jS < scaler; jS++)
                        {
                            bmp.SetPixel(i * scaler + iS, j * scaler + jS, color);
                        }
                    }
                }
            }
            bmp.Save($@"C:\Users\gabri\Desktop\Code Shit\TestFolder\{name}.png");
        }
    }
}
