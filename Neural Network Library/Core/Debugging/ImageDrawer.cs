using Neural_Network_Library.Core.Math;
using System.Drawing;

namespace Neural_Network_Library.Core.Debugging
{
    internal class ImageDrawer
    {
        private readonly string _path;
        private readonly int _scaler;

        internal ImageDrawer(int scaler, string path)
        {
            _path = path;
            _scaler = scaler;
        }

        internal void SaveFloatMatrixAsBitmap(string name, float[,] m, bool signedColors = false)
        {
            int mSizeX = m.GetLength(0);
            int mSizeY = m.GetLength(1);

            Bitmap bmp = new Bitmap(mSizeX * _scaler, mSizeY * _scaler);
            for (int i = 0; i < mSizeX; i++)
            {
                for (int j = 0; j < mSizeY; j++)
                {
                    Color color = GetValueColor(m[j, i], signedColors);
                    SetPixel(bmp, i, j, color);
                }
            }

            bmp.Save($"{_path}{name}.png");
        }

        private static Color GetValueColor(float val, bool signedColors)
        {
            if (!signedColors)
            {
                int intensity = (int)MathF.Min(val.Remap(0, 1, 0, 255), 255);

                return val switch
                {
                    float n when n is < 0 => Color.Black,
                    float n when n is >= 0 and <= 1 => Color.FromArgb(intensity, intensity, intensity),
                    float n when n is > 1 => Color.White,
                    _ => Color.Green
                };
            }
            else
            {
                int intensity = (int)MathF.Min(MathF.Abs(val).Remap(0, 1, 0, 255), 255);

                return val switch
                {
                    float n when n < 0 => Color.FromArgb(intensity, 0, 0),
                    float n when n > 0 => Color.FromArgb(0, 0, intensity),
                    _ => Color.Green
                };
            }
        }

        private void SetPixel(Bitmap bmp, int x, int y, Color color)
        {
            for (int i = 0; i < _scaler; i++)
            {
                for (int j = 0; j < _scaler; j++)
                {
                    int pX = x * _scaler + i;
                    int pY = y * _scaler + j;

                    bmp.SetPixel(pX, pY, color);
                }
            }
        }
    }
}
