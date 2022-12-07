using Neural_Network_Library;
using Neural_Network_Library.Backpropagation;
using static YggdrasilCore.Printer;

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
            NeuralNetwork network = new NeuralNetwork(layers);
            Backpropagation backpropagation = new Backpropagation(network, trainset);

            backpropagation.Run(500, 100, 0.1f);

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

                dataset[i] = new Datapoint(answer, data);
            }

            return dataset;
        }
    }
}
