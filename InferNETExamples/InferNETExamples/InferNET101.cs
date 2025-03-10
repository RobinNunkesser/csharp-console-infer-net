// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace InferNETExamples
{
    public static class InferNET101
    {
        static void Main()
        {
            Console.WriteLine("\n************");
            Console.WriteLine("CyclingTime1");
            Console.WriteLine("************\n");
            RunCyclingSamples.RunCyclingTime1();
            Console.Write("\nPress the spacebar to continue.");
            Console.ReadKey();

            Console.WriteLine("\n************");
            Console.WriteLine("CyclingTime2");
            Console.WriteLine("************\n");
            RunCyclingSamples.RunCyclingTime2();
            Console.Write("\nPress the spacebar to continue.");
            Console.ReadKey();

            Console.WriteLine("\n************");
            Console.WriteLine("CyclingTime3");
            Console.WriteLine("************\n");
            RunCyclingSamples.RunCyclingTime3();
            Console.Write("\nPress the spacebar to continue.");
            Console.ReadKey();

        }
    }
}