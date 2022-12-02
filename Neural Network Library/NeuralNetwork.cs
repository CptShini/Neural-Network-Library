namespace Neural_Network_Library
{
    public class NeuralNetwork
    {
        public float Fitness { get; set; }

        int[] layers;
        float[][] neurons;
        float[][][] weights;

        ActivationFunctionType _activationFunctionType = ActivationFunctionType.Sigmoid;

        #region Initialization

        public NeuralNetwork()
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
                weights[i] = new float[layers[i] + 1][]; // Since we also need a weight for the bias neuron to connect from.

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

        /// <summary>
        /// Randomizes the weights of the network to a value between -1f and 1f.s
        /// </summary>
        void RandomizeWeights()
        {
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        weights[i][j][k] = Rand(-1f, 1f);
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

            return neurons[layers.Length - 1];
        }

        float Activate(float val) => ActivationFunction.Activate(val, _activationFunctionType);

        #endregion

        #region Breeding

        public NeuralNetwork(NeuralNetwork p1, NeuralNetwork p2)
        {
            InitializeNetwork(p1.layers, _activationFunctionType);

            for (int i = 0; i < p1.weights.Length; i++)
            {
                for (int j = 0; j < p1.weights[i].Length; j++)
                {
                    for (int k = 0; k < p1.weights[i][j].Length; k++)
                    {
                        weights[i][j][k] = (Rand(0, 2) == 0) ? p1.weights[i][j][k] : p2.weights[i][j][k];
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
                        if (Rand(0, p) == 0) { weights[i][j][k] = Mutation(weights[i][j][k]); }
                    }
                }
            }
        }

        float Mutation(float val)
        {
            float result = val;

            int p = Rand(0, 4);

            switch (p)
            {
                case 0:
                    result += Rand(0f, 1f);
                    break;
                case 1:
                    result -= Rand(0f, 1f);
                    break;
                case 2:
                    result *= Rand(0.1f, 1f);
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

            result += "Fitness: " + Fitness + "\n\n";

            result += "Neurons:\n";

            for (int i = 0; i < layers.Length; i++)
            {
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    result += "[" + neurons[i][j] + "]";
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
                        result += "[" + weights[i][j][k].ToString("F2") + "]";
                    }
                    result += "\n";
                }
                result += "\n";
            }

            Console.WriteLine(result);
        }

        public static float Rand(float min, float max)
        {
            Random r = new Random();

            float val = (float)r.NextDouble();
            return Remap(val, 0, 1, min, max);
        }

        public static int Rand(int min, int max)
        {
            Random r = new Random();

            int val = r.Next(min, max);
            return val;
        }

        public static float Remap(float value, float from1, float to1, float from2, float to2) => ((value - from1) / (to1 - from1) * (to2 - from2)) + from2;

        #endregion
    }
}