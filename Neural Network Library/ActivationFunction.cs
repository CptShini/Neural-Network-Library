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

        internal static float DerivedActivate(float val, ActivationFunctionType type)
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

        internal static void Activate(float[] outputVector, float[] inputVector, ActivationFunctionType type)
        {
            for (int i = 0; i < outputVector.Length; i++)
            {
                outputVector[i] = Activate(inputVector[i], type);
            }
        }

        internal static void DerivedActivate(float[] outputVector, float[] inputVector, ActivationFunctionType type)
        {
            for (int i = 0; i < outputVector.Length; i++)
            {
                outputVector[i] = DerivedActivate(inputVector[i], type);
            }
        }

        internal static void Activate(float[,] outputMatrix, float[,] inputMatrix, ActivationFunctionType type)
        {
            for (int i = 0; i < outputMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < outputMatrix.GetLength(1); j++)
                {
                    outputMatrix[i, j] = Activate(inputMatrix[i, j], type);
                }
            }
        }

        internal static void DerivedActivate(float[,] outputMatrix, float[,] inputMatrix, ActivationFunctionType type)
        {
            for (int i = 0; i < outputMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < outputMatrix.GetLength(1); j++)
                {
                    outputMatrix[i, j] = DerivedActivate(inputMatrix[i, j], type);
                }
            }
        }
    }

    public enum ActivationFunctionType { Step, Linear, Sigmoid, Tanh, ReLU };
}