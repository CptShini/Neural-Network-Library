using Neural_Network_Library;
using Neural_Network_Library.Backpropagation;
using Neural_Network_Library.NetworkTypes;

namespace NumImgTest
{
    public class Program
    {
        private static void Main(string[] args)
        {
            Datapoint[] dataset = ImportDataset(@"C:\Users\gabri\Desktop\Code Shit\train.csv");
            Datapoint[] trainset = dataset[0..40000];
            Datapoint[] testset = dataset[40000..42000];

            int[] layers = { 784, 16, 16, 10 };
            ClassifierNetwork network = new ClassifierNetwork(layers);
            Backpropagation backpropagation = new Backpropagation(network, trainset, testset);

            backpropagation.Run(5000, 100, 1f, 200);

            foreach (Datapoint datapoint in testset)
            {
                network.Classify(datapoint.InputData);

                int answer = datapoint.DesiredOutput.ToList().IndexOf(datapoint.DesiredOutput.Max());
                int guess = network.Guess;
                float confidence = network.Confidence;

                Console.WriteLine($"{answer} | {guess} | {confidence * 100:00.00}%");
            }

            /*Bitmap[] bmp = new Bitmap[dataset.Length];
            for (int i = 0; i < 200; i++)
            {
                bmp[i] = new Bitmap(28, 28);
                for (int j = 1; j < dataset[i].Length; j++)
                {
                    int val = dataset[i][j];

                    int x = (j - 1) % 28;
                    int y = (j - 1) / 28;
                    Color color = Color.FromArgb(val, val, val);

                    bmp[i].SetPixel(x, y, color);
                }
                bmp[i].Save($@"C:\Users\gabri\Desktop\TestFolder\{dataset[i][0]}-{DateTime.Now.Ticks}.png");
            }*/
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
