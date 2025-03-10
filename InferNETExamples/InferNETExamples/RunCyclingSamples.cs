// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace InferNETExamples
{
    public static class RunCyclingSamples
    {
        public static void RunCyclingTime1()
        {
            // [1] The model
            var averageTime = Variable.GaussianFromMeanAndPrecision(15, 0.01);
            var trafficNoise = Variable.GammaFromShapeAndScale(2.0, 0.5);

            var travelTimeMonday = Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise);
            var travelTimeTuesday = Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise);
            var travelTimeWednesday = Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise);

            // [2] Train the model
            travelTimeMonday.ObservedValue = 13;
            travelTimeTuesday.ObservedValue = 17;
            travelTimeWednesday.ObservedValue = 16;

            var engine = new InferenceEngine();

            var averageTimePosterior = engine.Infer<Gaussian>(averageTime);
            var trafficNoisePosterior = engine.Infer<Gamma>(trafficNoise);

            Console.WriteLine("averageTimePosterior: " + averageTimePosterior);
            Console.WriteLine("trafficNoisePosterior: " + trafficNoisePosterior);

            // [3] Add a prediction variable and retrain the model
            var tomorrowsTime = Variable.GaussianFromMeanAndPrecision(
                averageTime,
                trafficNoise);

            var tomorrowsTimeDist = engine.Infer<Gaussian>(tomorrowsTime);
            var tomorrowsMean = tomorrowsTimeDist.GetMean();
            var tomorrowsStdDev = Math.Sqrt(tomorrowsTimeDist.GetVariance());

            // Write out the results.
            Console.WriteLine("Tomorrows predicted time: {0:f2} plus or minus {1:f2}", tomorrowsMean, tomorrowsStdDev);

            // You can also ask other questions of the model 
            double probTripTakesLessThan18Minutes = engine.Infer<Bernoulli>(tomorrowsTime < 18.0).GetProbTrue();
            Console.WriteLine(
                "Probability that the trip takes less than 18 min: {0:f2}",
                probTripTakesLessThan18Minutes);
        }

        public static void RunCyclingTime2()
        {
            double[] trainingData = new double[] { 13, 17, 16, 12, 13, 12, 14, 18, 16, 16 };
            ModelData initPriors = new ModelData(
                Gaussian.FromMeanAndPrecision(1.0, 0.01),
                Gamma.FromShapeAndScale(2.0, 0.5));

            // Train the model
            CyclistTraining cyclistTraining = new CyclistTraining();
            cyclistTraining.CreateModel();
            cyclistTraining.SetModelData(initPriors);

            ModelData posteriors1 = cyclistTraining.InferModelData(trainingData);
            Console.WriteLine("Average travel time = " + posteriors1.AverageTimeDist);
            Console.WriteLine("Traffic noise = " + posteriors1.TrafficNoiseDist);

            // Make predictions based on the trained model
            CyclistPrediction cyclistPrediction = new CyclistPrediction();
            cyclistPrediction.CreateModel();
            cyclistPrediction.SetModelData(posteriors1);

            Gaussian tomorrowsTimeDist = cyclistPrediction.InferTomorrowsTime();

            double tomorrowsMean = tomorrowsTimeDist.GetMean();
            double tomorrowsStdDev = Math.Sqrt(tomorrowsTimeDist.GetVariance());

            Console.WriteLine("Tomorrows average time: {0:f2}", tomorrowsMean);
            Console.WriteLine("Tomorrows standard deviation: {0:f2}", tomorrowsStdDev);
            Console.WriteLine("Probability that tomorrow's time is < 18 min: {0}",
                              cyclistPrediction.InferProbabilityTimeLessThan(18.0));

            // Second round of training
            double[] trainingData2 = new double[] { 17, 19, 18, 21, 15 };

            cyclistTraining.SetModelData(posteriors1);
            ModelData posteriors2 = cyclistTraining.InferModelData(trainingData2);

            Console.WriteLine("\n2nd training pass");
            Console.WriteLine("Average travel time = " + posteriors2.AverageTimeDist);
            Console.WriteLine("Traffic noise = " + posteriors2.TrafficNoiseDist);

            // Predictions based on two rounds of training
            cyclistPrediction.SetModelData(posteriors2);

            tomorrowsTimeDist = cyclistPrediction.InferTomorrowsTime();
            tomorrowsMean = tomorrowsTimeDist.GetMean();
            tomorrowsStdDev = Math.Sqrt(tomorrowsTimeDist.GetVariance());

            Console.WriteLine("Tomorrows average time: {0:f2}", tomorrowsMean);
            Console.WriteLine("Tomorrows standard deviation: {0:f2}", tomorrowsStdDev);
            Console.WriteLine("Probability that tomorrow's time is < 18 min: {0}",
                              cyclistPrediction.InferProbabilityTimeLessThan(18));
        }

        public static void RunCyclingTime3()
        {
            ModelDataMixed initPriors;

            double[] trainingData = new double[] { 13, 17, 16, 12, 13, 12, 14, 18, 16, 16, 27, 32 };
            initPriors.AverageTimeDist = new Gaussian[] { new Gaussian(15.0, 100), new Gaussian(30.0, 100) };
            initPriors.TrafficNoiseDist = new Gamma[] { new Gamma(2.0, 0.5), new Gamma(2.0, 0.5) };
            initPriors.MixingDist = new Dirichlet(1, 1);

            CyclistMixedTraining cyclistMixedTraining = new CyclistMixedTraining();
            cyclistMixedTraining.CreateModel();
            cyclistMixedTraining.SetModelData(initPriors);

            ModelDataMixed posteriors = cyclistMixedTraining.InferModelData(trainingData);

            Console.WriteLine("Average time distribution 1 = " + posteriors.AverageTimeDist[0]);
            Console.WriteLine("Average time distribution 2 = " + posteriors.AverageTimeDist[1]);
            Console.WriteLine("Noise distribution 1 = " + posteriors.TrafficNoiseDist[0]);
            Console.WriteLine("Noise distribution 2 = " + posteriors.TrafficNoiseDist[1]);
            Console.WriteLine("Mixing coefficient distribution = " + posteriors.MixingDist);

            CyclistMixedPrediction cyclistMixedPrediction = new CyclistMixedPrediction();
            cyclistMixedPrediction.CreateModel();
            cyclistMixedPrediction.SetModelData(posteriors);

            Gaussian tomorrowsTime = cyclistMixedPrediction.InferTomorrowsTime();

            double tomorrowsMean = tomorrowsTime.GetMean();
            double tomorrowsStdDev = Math.Sqrt(tomorrowsTime.GetVariance());

            Console.WriteLine("Tomorrows expected time: {0:f2}", tomorrowsMean);
            Console.WriteLine("Tomorrows standard deviation: {0:f2}", tomorrowsStdDev);
        }

        
    }
}