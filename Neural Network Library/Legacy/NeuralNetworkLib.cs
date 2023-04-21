using Neural_Network_Library.Core;
using Random = Neural_Network_Library.Core.Random;

namespace Neural_Network_Library.Legacy
{
    public class NeuralNetworkLib
    {
        public float Fitness { get; set; }

        int[] layers;
        float[][] neurons;
        float[][][] weights;

        ActivationFunctionType _activationFunctionType = ActivationFunctionType.Sigmoid;

        float[][] neuronSens;
        float[][][] weightSens;

        #region Initialization

        public NeuralNetworkLib()
        {

        }

        /// <summary>
        /// Initializes the network with the given network structure in the layer parameter.
        /// </summary>
        /// <param name="_layers">The structure of the network as an int array; i.e. {784, 16, 16, 10} giving a structure of 784 input neurons, 2x16 hidden layers, and an output layer of length 10.</param>
        public void InitializeNetwork(int[] _layers, ActivationFunctionType _activationType)
        {
            layers = _layers;
            InitializeNeurons();
            InitializeWeights();
            RandomizeWeights();

            _activationFunctionType = _activationType;

            neuronSens = new float[neurons.Length][];
            weightSens = new float[weights.Length][][];

            for (int i = 0; i < neurons.Length; i++)
            {
                neuronSens[i] = new float[neurons[i].Length];
            }

            weightSens = CreateWeightArray();
        }

        /// <summary>
        /// Initializes the neuron jagged array to the corrosponding length given by layer field; also adds bias neurons.
        /// </summary>
        void InitializeNeurons()
        {
            neurons = new float[layers.Length][];

            for (int i = 0; i < layers.Length; i++)
            {
                bool lastLayer = i >= layers.Length - 1;
                neurons[i] = !lastLayer ? new float[layers[i] + 1] : new float[layers[i]]; // Adds an extra neuron, which is the bias neuron, if we aren't in the last layer.

                for (int j = 0; j < neurons[i].Length; j++)
                {
                    if (!lastLayer) // Since we don't want to give the neurons in the output layer a value, we do this if statement.
                    {
                        bool biasNeuron = j == neurons[i].Length - 1; // Checks if this is the last neuron in the list, aka. the bias neuron.
                        neurons[i][j] = !biasNeuron ? 0f : 1f; // Initializes the neurons value to 0 if a regular neuron, and 1 for bias neurons.
                    }
                }
            }
        }

        /// <summary>
        /// Initializes weight jagged array.
        /// </summary>
        void InitializeWeights()
        {
            weights = new float[layers.Length - 1][][]; // Makes it layers.length - 1 since our output layer doesn't connect anywhere.

            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = new float[neurons[i].Length][];

                for (int j = 0; j < weights[i].Length; j++)
                {
                    weights[i][j] = new float[layers[i + 1]]; // Dependent on the number of neurons in next layer.

                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        weights[i][j][k] = 0f;
                    }
                }
            }
        }

        public float[][][] CreateWeightArray()
        {
            float[][][] result = new float[layers.Length - 1][][]; // Makes it layers.length - 1 since our output layer doesn't connect anywhere.

            for (int i = 0; i < weights.Length; i++)
            {
                result[i] = new float[neurons[i].Length][];

                for (int j = 0; j < weights[i].Length; j++)
                {
                    result[i][j] = new float[layers[i + 1]]; // Dependent on the number of neurons in next layer.

                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        result[i][j][k] = 0f;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Randomizes the weights of the network to a value between -1f and 1f.
        /// </summary>
        void RandomizeWeights()
        {
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        weights[i][j][k] = Random.Range(-1f, 1f);
                    }
                }
            }
        }

        #endregion

        #region FeedForward

        public float[] FeedForward(float[] inputs)
        {
            // Copies the inputs to the network input layer.
            for (int i = 0; i < layers[0]; i++)
            {
                neurons[0][i] = inputs[i];
            }

            // Propagates the values through the network.
            for (int i = 1; i < layers.Length; i++)
            {
                bool lastLayer = i >= layers.Length - 1;

                for (int j = 0; j < neurons[i].Length - (!lastLayer ? 1 : 0); j++) // We do the subtraction part to not attempt to compute the activation of the bias neurons.
                {
                    for (int k = 0; k < neurons[i - 1].Length; k++)
                    {
                        neurons[i][j] += neurons[i - 1][k] * weights[i - 1][k][j];
                    }

                    neurons[i][j] = Activate(neurons[i][j]);
                }
            }

            return neurons[^1]; // Returns the output layer.
        }

        public (int, float) Classify()
        {
            float outputSum = neurons[^1].Sum();

            // Normalizes the outputs; making them sum = 1.
            for (int j = 0; j < layers[^1]; j++)
            {
                neurons[^1][j] /= outputSum;
            }

            float networkCertainty = neurons[^1].Max();
            int networkAnswer = neurons[^1].ToList().IndexOf(networkCertainty);

            return (networkAnswer, networkCertainty);
        }

        public float Cost(float[] AnswerSheet)
        {
            float costSum = 0f;

            for (int i = 0; i < layers[^1]; i++)
            {
                float a = neurons[^1][i];
                float y = AnswerSheet[i];
                float cost = y - a;
                cost *= cost;

                costSum += cost;
            }

            return costSum;
        }

        public float[][][] Backpropagate(float[] AnswerSheet)
        {
            for (int i = layers.Length - 1; i >= 1; i--)
            {
                bool lastLayer = i >= layers.Length - 1;

                for (int j = 0; j < neurons[i].Length - (lastLayer ? 0 : 1); j++)
                {
                    if (lastLayer)
                    {
                        float a = neurons[i][j];
                        float y = AnswerSheet[j];
                        int n_L = layers[^1];

                        neuronSens[i][j] = 2f * (y - a) / n_L;
                    }
                    else
                    {
                        float sensSum = 0f;
                        for (int k = 0; k < neurons[i + 1].Length - 1; k++)
                        {
                            float w = weights[i][j][k];
                            float neuronZ = MathF.Log(neurons[i + 1][k] / (1 - neurons[i + 1][k]));
                            float derivedZ = derivedActivate(neuronZ);

                            sensSum += w * derivedZ * neuronSens[i + 1][k];
                        }

                        neuronSens[i][j] = sensSum;
                    }

                    float z = MathF.Log(neurons[i][j] / (1 - neurons[i][j]));
                    float sig = derivedActivate(z);
                    for (int k = 0; k < neurons[i - 1].Length; k++)
                    {
                        float a = neurons[i - 1][k];
                        weightSens[i - 1][k][j] = a * sig * neuronSens[i][j];
                    }
                }
            }

            float derivedActivate(float x) => Activate(x) * (1 - Activate(x));

            return weightSens;
        }

        public static float[][][] AddGradientVectors(float[][][] v1, float[][][] v2)
        {
            for (int i = 0; i < v1.Length; i++)
            {
                for (int j = 0; j < v1[i].Length; j++)
                {
                    for (int k = 0; k < v1[i][j].Length; k++)
                    {
                        v1[i][j][k] += v2[i][j][k];
                    }
                }
            }

            return v1;
        }

        public static float[][][] DivideGradientVector(float[][][] v, float x)
        {
            for (int i = 0; i < v.Length; i++)
            {
                for (int j = 0; j < v[i].Length; j++)
                {
                    for (int k = 0; k < v[i][j].Length; k++)
                    {
                        v[i][j][k] /= x;
                    }
                }
            }

            return v;
        }

        public void ApplyGradientVector(float[][][] gradVector)
        {
            float step = 0.001f;
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        weights[i][j][k] -= step * gradVector[i][j][k];
                    }
                }
            }
        }

        float Activate(float val) => ActivationFunction.Activate(val, _activationFunctionType);

        #endregion

        #region Breeding

        public NeuralNetworkLib(NeuralNetworkLib p1, NeuralNetworkLib p2)
        {
            InitializeNetwork(p1.layers, _activationFunctionType);

            for (int i = 0; i < p1.weights.Length; i++)
            {
                for (int j = 0; j < p1.weights[i].Length; j++)
                {
                    for (int k = 0; k < p1.weights[i][j].Length; k++)
                    {
                        weights[i][j][k] = Random.Range(0, 2) == 0 ? p1.weights[i][j][k] : p2.weights[i][j][k];
                    }
                }
            }

            MutateNetwork(15);
        }

        #region Mutation

        public void MutateNetwork(int p)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        if (Random.Range(0, p) == 0) { weights[i][j][k] = Mutation(weights[i][j][k]); }
                    }
                }
            }
        }

        float Mutation(float val)
        {
            float result = val;

            int p = Random.Range(0, 4);

            switch (p)
            {
                case 0:
                    result += Random.Range(0f, 1f);
                    break;
                case 1:
                    result -= Random.Range(0f, 1f);
                    break;
                case 2:
                    result *= Random.Range(0.1f, 1f);
                    break;
                case 3:
                    result *= -1;
                    break;
            }
            return Math.Clamp(result, -1f, 1f);
        }

        #endregion

        #endregion

        #region Utility

        public void PrintNetwork()
        {
            string result = "";

            result += $"Fitness: {Fitness}\n\n";

            result += "Neurons:\n";

            for (int i = 0; i < layers.Length; i++)
            {
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    result += $"[{(i < layers.Length - 1 ? $"{neurons[i][j]:0.00}" : $"{neurons[i][j] * 100:00.0}%")}]";
                }
                result += "\n";
            }

            result += "\n\n";

            result += "Weights:\n";

            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        result += $"[{weights[i][j][k]:0.00}]";
                    }
                    result += "\n";
                }
                result += "\n";
            }

            Console.WriteLine(result);
        }

        public static float Remap(float value, float from1, float to1, float from2, float to2) => (value - from1) / (to1 - from1) * (to2 - from2) + from2;

        #endregion
    }
}