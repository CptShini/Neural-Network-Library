using Neural_Network_Library.Core;
using System.Linq;
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

            float maxVal = signedColors ? 1f : m.Max();
            float minVal = signedColors ? -1f : m.Min();

            Bitmap bmp = new Bitmap(mSizeX * _scaler, mSizeY * _scaler);
            for (int i = 0; i < mSizeX; i++)
            {
                for (int j = 0; j < mSizeY; j++)
                {
                    float val = m[j, i];
                    Color color = GetColor(val, minVal, maxVal, signedColors);
                    SetPixel(bmp, i, j, color);
                }
            }

            bmp.Save($"{_path}{name}.png");
        }

        private Color GetColor(float val, float minVal, float maxVal, bool signedColors) => signedColors ? GetSignedColor(val) : GetUnsignedColor(val, minVal, maxVal);

        private Color GetUnsignedColor(float val, float minVal, float maxVal)
        {
            int intensity = (int)val.Remap(minVal, maxVal, 0, 255);
            return Color.FromArgb(intensity, intensity, intensity);
        }

        private Color GetSignedColor(float val)
        {
            int intensity = (int)MathF.Abs(val).Remap(0f, 1f, 0f, 255f);

            return val switch
            {
                float n when n < 0 => Color.FromArgb(intensity, 0, 0),
                float n when n > 0 => Color.FromArgb(0, 0, intensity),
                _ => Color.Green
            };
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