namespace Neural_Network_Library.Backpropagation
{
    public class Datapoint
    {
        public float[] DesiredOutput;
        public float[] InputData;

        public Datapoint(float[] inputData, float[] desiredOutput)
        {
            InputData = inputData;
            DesiredOutput = desiredOutput;
        }
    }
}