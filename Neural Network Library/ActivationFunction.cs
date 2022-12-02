using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_Library
{
    public static class ActivationFunction
    {
        public static float Activate(float val, ActivationFunctionType type)
        {
            return type switch
            {
                ActivationFunctionType.Linear => val,
                ActivationFunctionType.Step => val < 0 ? 0 : 1,
                ActivationFunctionType.Sigmoid => 1 / (1 + MathF.Exp(-val)),
                ActivationFunctionType.Tanh => MathF.Tanh(val),
                ActivationFunctionType.ReLU => MathF.Max(0, val),
                _ => 1 / (1 + MathF.Exp(-val))
            };
        }
    }

    public enum ActivationFunctionType { Step, Linear, Sigmoid, Tanh, ReLU };
}
