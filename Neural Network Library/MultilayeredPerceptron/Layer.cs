using Neural_Network_Library.Core;
using static Neural_Network_Library.Core.NeuralNetworkMath;
using Random = Neural_Network_Library.Core.Random;

namespace Neural_Network_Library.MultilayeredPerceptron
{
    internal class Layer
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
                    _w[j, k] = Random.Range(-1f, 1f);
                }

                _b[j] = Random.Range(-1f, 1f);
            }
        }

        internal float[] FeedForward(float[] input)
        {
            _a_1 = input;
            
            MatrixVectorProduct(_z, _w, _a_1);
            AddVectors(_z, _z, _b);
            Activate(_a, _z);
            
            return _a;
        }

        private void Activate(float[] outputVector, float[] inputVector) => ActivationFunction.Activate(outputVector, inputVector, _activationFunctionType);
    }
}