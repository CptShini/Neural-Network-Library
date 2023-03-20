using System.Runtime.InteropServices;

namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    public class ConvolutionalNeuralNetwork
    {
        private readonly float[,] _kernel;
        private readonly int _kernelSize;
        private readonly int _kernelSizeDelta;

        public ConvolutionalNeuralNetwork(float[,] kernel)
        {
            _kernel = kernel;
            _kernelSize = kernel.GetLength(0);
            _kernelSizeDelta = (int)(_kernelSize / 2 - 0.5f);
        }

        public float[,] Convolve(float[,] input)
        {
            int inputWidth = input.GetLength(0);
            int inputHeight = input.GetLength(1);

            float[,] convolution = new float[inputWidth, inputHeight];

            for (int x = 0; x < inputWidth; x++)
            {
                for (int y = 0; y < inputHeight; y++)
                {
                    int cellsScanned = 0;
                    float convolutionSum = 0f;

                    for (int i = 0; i < _kernelSize; i++)
                    {
                        for (int j = 0; j < _kernelSize; j++)
                        {
                            int offsetX = i - _kernelSizeDelta;
                            int offsetY = j - _kernelSizeDelta;

                            int pixelX = x + offsetX;
                            int pixelY = y + offsetY;

                            if (pixelX < 0 || pixelY < 0 || pixelX >= inputWidth || pixelY >= inputHeight) continue;

                            convolutionSum += input[pixelX, pixelY] * _kernel[i, j];
                            cellsScanned++;
                        }
                    }

                    convolution[x, y] = convolutionSum / cellsScanned;
                }
            }

            return convolution;
        }
    }
}
