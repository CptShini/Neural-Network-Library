namespace Neural_Network_Library.MultilayeredPerceptron.Backpropagation
{
    public class Datapoint
    {
        public readonly float[] DesiredOutput;
        public readonly float[] InputData;

        public Datapoint(float[] inputData, float[] desiredOutput)
        {
            InputData = inputData;
            DesiredOutput = desiredOutput;
        }
    }
}