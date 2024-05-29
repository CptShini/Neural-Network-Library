using Neural_Network_Library.Core;
using Neural_Network_Library.Interfaces.MLP;
using static Neural_Network_Library.Core.RandomNumberGenerator;

namespace Neural_Network_Library.Networks.MLP;

internal class MLPLayer : ILayer
{
    internal readonly float[] _a, _z, _b;
    internal readonly float[,] _w;
    internal float[] _a_1;

    internal readonly ActivationFunctionType _activationFunctionType;

    private readonly int _inputSize, _outputSize;

    internal MLPLayer(int inputSize, int outputSize, ActivationFunctionType activationFunctionType)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;

        _a = new float[outputSize];
        _z = new float[outputSize];
        _b = new float[outputSize];
        _w = new float[outputSize, inputSize];

        _a_1 = Array.Empty<float>();

        _activationFunctionType = activationFunctionType;

        InitializeWeightsAndBiases();
    }

    private void InitializeWeightsAndBiases()
    {
        for (int j = 0; j < _outputSize; j++)
        {
            for (int k = 0; k < _inputSize; k++)
            {
                _w[j, k] = RandomRange(-1f, 1f);
            }

            _b[j] = RandomRange(-1f, 1f);
        }
    }

    public float[] FeedForward(float[] input)
    {
        _a_1 = input;

        for (int j = 0; j < _outputSize; j++)
        {
            _z[j] = _b[j];

            for (int k = 0; k < _inputSize; k++)
            {
                _z[j] += _w[j, k] * _a_1[k];
            }

            _a[j] = ActivationFunction.Activate(_z[j], _activationFunctionType);

        }
        
        return _a;
    }
}