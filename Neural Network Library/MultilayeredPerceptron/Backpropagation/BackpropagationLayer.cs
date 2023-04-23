using Neural_Network_Library.Core;
using static Neural_Network_Library.Core.ActivationFunction;

namespace Neural_Network_Library.MultilayeredPerceptron.Backpropagation
{
    internal abstract class BackpropagationLayer
    {
        private protected readonly Layer _layer;

        private readonly float[] _da, _dz, _db;
        private readonly float[,] _dw;

        private int _epochCount;

        private protected BackpropagationLayer(Layer layer)
        {
            _layer = layer;

            _da = new float[layer._a.Length];
            _dz = new float[layer._z.Length];
            _db = new float[layer._b.Length];
            _dw = new float[layer._w.GetLength(0), layer._w.GetLength(1)];

            _epochCount = 0;
        }

        internal void SumToGradientVector()
        {
            for (int j = 0; j < _da.Length; j++)
            {
                _da[j] = Calculate_da(j);
                _dz[j] = DerivedActivate(_layer._z[j]) * _da[j];

                _db[j] += _dz[j];

                for (int k = 0; k < _dw.GetLength(1); k++)
                {
                    _dw[j, k] += _layer._a_1[k] * _dz[j];
                }
            }

            _epochCount++;
        }

        internal void ApplyGradientVector(float learnRate)
        {
            for (int j = 0; j < _layer._b.Length; j++)
            {
                for (int k = 0; k < _layer._w.GetLength(1); k++)
                {
                    _layer._w[j, k] -= _dw[j, k] / _epochCount * learnRate;
                    _dw[j, k] = 0f;
                }

                _layer._b[j] -= _db[j] / _epochCount * learnRate;
                _db[j] = 0f;
            }
            
            _epochCount = 0;
        }

        private protected abstract float Calculate_da(int k);

        internal float Calculate_NextLayer_da(int k)
        {
            float sum = 0f;
            for (int j = 0; j < _da.Length; j++)
            {
                sum += _layer._w[j, k] * _dz[j];
            }

            return sum;
        }

        private float DerivedActivate(float val) => val.Activate(_layer._activationFunctionType, true);
    }
}