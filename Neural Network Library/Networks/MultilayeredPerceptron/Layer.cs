using Neural_Network_Library.Core;
using static Neural_Network_Library.Core.RandomNumberGenerator;

namespace Neural_Network_Library.Networks.MultilayeredPerceptron
{
    internal class Layer : ILayer
    {
        internal readonly float[] _a, _z, _b;
        internal readonly float[,] _w;
        internal float[] _a_1;

        internal readonly ActivationFunctionType _activationFunctionType;

        internal Layer(int inputSize, int outputSize, ActivationFunctionType activationFunctionType)
        {
            _a = new float[outputSize];
            _z = new float[outputSize];
            _b = new float[outputSize];
            _w = new float[outputSize, inputSize];

            _a_1 = new float[inputSize];

            _activationFunctionType = activationFunctionType;

            InitializeWeightsAndBiases();
        }

        private void InitializeWeightsAndBiases()
        {
            for (int j = 0; j < _w.GetLength(0); j++)
            {
                for (int k = 0; k < _w.GetLength(1); k++)
                {
                    _w[j, k] = RandomRange(-1f, 1f);
                }

                _b[j] = RandomRange(-1f, 1f);
            }
        }

        float[] ILayer.FeedForward(float[] input)
        {
            _a_1 = input;

            for (int j = 0; j < _w.GetLength(0); j++)
            {
                _z[j] = _b[j];

                for (int k = 0; k < _w.GetLength(1); k++)
                {
                    _z[j] += _w[j, k] * _a_1[k];
                }

                _a[j] = ActivationFunction.Activate(_z[j], _activationFunctionType);

            }
            
            return _a;
        }
    }
}