namespace Neural_Network_Library.Networks.ConvolutionalNeuralNetwork
{
    internal struct Tensor
    {
        private readonly float[][,] _tensor;

        internal int Size { get; private set; }

        internal int Depth { get; }

        internal int Volume => Depth * Size * Size;

        internal Tensor(float[,] matrix)
        {
            Size = matrix.GetLength(0);
            Depth = 1;

            _tensor = new float[][,] { matrix };
        }

        internal Tensor(int depth)
        {
            Size = -1;
            Depth = depth;

            _tensor = new float[depth][,];
        }

        internal Tensor(int depth, int size)
        {
            Size = size;
            Depth = depth;

            _tensor = new float[depth][,];
            for (int d = 0; d < depth; d++)
            {
                _tensor[d] = new float[size, size];
            }
        }

        internal float[,] this[int d]
        {
            get => _tensor[d];
            set
            {
                Size = value.GetLength(0);
                _tensor[d] = value;
            }
        }

        internal float this[int d, int x, int y]
        {
            get => _tensor[d][x, y];
            set => _tensor[d][x, y] = value;
        }
    }
}