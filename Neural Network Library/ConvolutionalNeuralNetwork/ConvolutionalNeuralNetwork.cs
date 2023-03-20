namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    public class ConvolutionalNeuralNetwork
    {
        private readonly float[,] _kernel;
        private readonly int _kernelSize;
        private readonly int _kernelSizeDelta;

        private int _inputWidth;
        private int _inputHeight;

        public ConvolutionalNeuralNetwork(float[,] kernel)
        {
            _kernel = kernel;
            _kernelSize = kernel.GetLength(0);
            _kernelSizeDelta = (int)((_kernelSize / 2) - 0.5f);
        }

        public float[,] Convolve(float[,] input)
        {
            _inputWidth = input.GetLength(0);
            _inputHeight = input.GetLength(1);

            float[,] convolution = new float[_inputWidth, _inputHeight];

            for (int x = 0; x < _inputWidth; x++)
            {
                for (int y = 0; y < _inputHeight; y++)
                {
                    Tuple<float, int> convolutionResult = ConvolveAt(x, y, input);
                    convolution[x, y] = convolutionResult.Item1 / convolutionResult.Item2;
                }
            }

            return convolution;
        }

        private Tuple<float, int> ConvolveAt(int x, int y, float[,] input)
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

                    if (OutsideBounds(pixelX, pixelY)) continue;

                    convolutionSum += input[pixelX, pixelY] * _kernel[i, j];
                    cellsScanned++;
                }
            }

            return new Tuple<float, int>(convolutionSum, cellsScanned);
        }

        private bool OutsideBounds(int pixelX, int pixelY) => pixelX < 0 || pixelY < 0 || pixelX >= _inputWidth || pixelY >= _inputHeight;
    }
}
