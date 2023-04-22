namespace Neural_Network_Library.Core.Debugging
{
    internal static class Printer
    {
        internal static void Print() => Console.Write("");

        internal static void PrintLine() => Console.WriteLine();

        internal static void Print<T>(T message) => Console.Write(message);

        internal static void PrintLine<T>(T message) => Console.WriteLine(message);

        internal static void PrintEnumerable<T>(IEnumerable<T> messages)
        {
            foreach (T message in messages)
            {
                Console.Write(message);
            }
        }

        internal static void PrintLineEnumerable<T>(IEnumerable<T> messages)
        {
            foreach (T message in messages)
            {
                Console.WriteLine(message);
            }
        }
    }
}
