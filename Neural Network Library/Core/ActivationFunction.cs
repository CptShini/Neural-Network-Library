using static Neural_Network_Library.Core.ActivationFunctionType;

namespace Neural_Network_Library.Core
{
    internal static class ActivationFunction
    {
        private static float Activate(this float val, ActivationFunctionType type)
        {
            return type switch
            {
                Linear => val,
                Step => val <= 0f ? 0f : 1f,
                Sigmoid => 1f / (1f + MathF.Exp(-val)),
                Tanh => MathF.Tanh(val),
                ReLU => val <= 0f ? 0f : val,
                _ => throw new ArgumentException()
            };
        }

        private static float DerivedActivate(this float val, ActivationFunctionType type)
        {
            return type switch
            {
                Linear => 1f,
                Step => 0f,
                Sigmoid => MathF.Exp(-val) / MathF.Pow(1 + MathF.Exp(-val), 2f),
                Tanh => 1f - MathF.Pow(MathF.Tanh(val), 2f),
                ReLU => val > 0f ? 1f : 0f,
                _ => throw new ArgumentException()
            };
        }

        internal static float Activate(this float val, ActivationFunctionType type, bool useDerivedActivation = false) =>
            !useDerivedActivation ? val.Activate(type) : val.DerivedActivate(type);
    }

    public enum ActivationFunctionType { Step, Linear, Sigmoid, Tanh, ReLU };
}