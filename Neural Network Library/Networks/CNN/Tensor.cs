using System.Drawing.Drawing2D;

namespace Neural_Network_Library.Networks.CNN;

internal struct Tensor
{
    private readonly float[][,] _tensor;

    internal int Size { get; private set; }

    internal int Depth { get; }

    internal int Volume => Depth * Size * Size;

    internal Tensor(float[,] matrix)
    {
        int width = matrix.GetLength(0);
        int height = matrix.GetLength(1);
        if (width != height) throw new ArgumentOutOfRangeException($"The given input matrix is not square, and is therefore not valid. Width is {width}, and height is {height}.");

        Size = width;
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