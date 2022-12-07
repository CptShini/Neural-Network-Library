using static Neural_Network_Library.ActivationFunctionType;

namespace Neural_Network_Library
{
    internal static class ActivationFunction
    {
        internal static float Activate(float val, ActivationFunctionType type)
        {
            return type switch
            {
                Linear => val,
                Step => val < 0f ? 0f : 1f,
                Sigmoid => 1f / (1f + MathF.Exp(-val)),
                Tanh => MathF.Tanh(val),
                ReLU => val < 0f ? 0f : val,
                _ => throw new NotImplementedException()
            };
        }

        internal static float DerivedActive(float val, ActivationFunctionType type)
        {
            return type switch
            {
                Linear => 1f,
                Step => 0f,
                Sigmoid => MathF.Exp(-val) / MathF.Pow(1 + MathF.Exp(-val), 2f),
                Tanh => 1f - MathF.Pow(MathF.Tanh(val), 2f),
                ReLU => val > 0f ? 1f : 0f,
                _ => throw new NotImplementedException()
            };
        }
    }

    public enum ActivationFunctionType { Step, Linear, Sigmoid, Tanh, ReLU };
}