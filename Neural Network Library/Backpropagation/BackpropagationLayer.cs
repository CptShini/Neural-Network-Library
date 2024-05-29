using Neural_Network_Library.Core;
using Neural_Network_Library.Networks.MLP;

namespace Neural_Network_Library.Backpropagation;

internal abstract class BackpropagationLayer
{
    private protected readonly MLPLayer _layer;

    private readonly float[] _da, _dz, _db;
    private readonly float[,] _dw;

    private int _epochCount;
    private readonly int _inputSize, _outputSize;

    private protected BackpropagationLayer(MLPLayer layer)
    {
        _layer = layer;
        _inputSize = layer._w.GetLength(1);
        _outputSize = layer._w.GetLength(0);

        _da = new float[_outputSize];
        _dz = new float[_outputSize];
        _db = new float[_outputSize];
        _dw = new float[_outputSize, _inputSize];

        _epochCount = 0;
    }

    internal void SumToGradientVector()
    {
        for (int j = 0; j < _outputSize; j++)
        {
            _da[j] = Calculate_da(j);
            _dz[j] = DerivedActivate(_layer._z[j]) * _da[j];

            _db[j] += _dz[j];

            for (int k = 0; k < _inputSize; k++)
            {
                _dw[j, k] += _layer._a_1[k] * _dz[j];
            }
        }

        _epochCount++;
    }

    internal void ApplyGradientVector(float learnRate)
    {
        for (int j = 0; j < _outputSize; j++)
        {
            for (int k = 0; k < _inputSize; k++)
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
        for (int j = 0; j < _outputSize; j++)
        {
            sum += _layer._w[j, k] * _dz[j];
        }

        return sum;
    }

    private float DerivedActivate(float val) => val.Activate(_layer._activationFunctionType, true);
}