using Neural_Network_Library.Core;
using Neural_Network_Library.Core.Math;
using Random = Neural_Network_Library.Core.Random;

namespace Neural_Network_Library.ConvolutionalNeuralNetwork
{
    internal class Kernel
    {
        private Tensor _filter;
        private float _bias;

        private readonly ActivationFunctionType _activationFunctionType;

        internal Kernel(int kernelDepth, int kernelSize, ActivationFunctionType activationFunctionType)
        {
            _activationFunctionType = activationFunctionType;

            _filter = new Tensor(kernelDepth, kernelSize);
            InitializeFiltersAndBiasTest();
        }

        private void InitializeFiltersAndBias()
        {
            for (int d = 0; d < _filter.Depth; d++)
            {
                for (int x = 0; x < _filter.Size; x++)
                {
                    for (int y = 0; y < _filter.Size; y++)
                    {
                        _filter[d, x, y] = Random.Range(-1f, 1f);
                    }
                }
            }

            _bias = Random.Range(-1f, 1f);
        }

        private void InitializeFiltersAndBiasTest()
        {
            long t = DateTime.Now.Ticks;

            Core.Debugging.ImageDrawer imageDrawer = new Core.Debugging.ImageDrawer(10, @"C:\Users\gabri\Desktop\Code Shit\TestFolder\");

            for (int d = 0; d < _filter.Depth; d++)
            {
                for (int x = 0; x < _filter.Size; x++)
                {
                    for (int y = 0; y < _filter.Size; y++)
                    {
                        _filter[d, x, y] = Random.Range(-1f, 1f);
                    }
                }
                imageDrawer.SaveFloatMatrixAsBitmap($"{t} Filter {d}", _filter[d], true);
            }

            _bias = Random.Range(-1f, 1f);
        }

        internal float[,] Convolve(Tensor input)
        {
            float[,] convolution = input.Convolve(_filter);

            convolution.AddToMatrix(_bias);
            convolution.Activate(_activationFunctionType);

            return convolution;
        }
    }
}