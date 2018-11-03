package opt.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.feedfwd.FeedForwardNetwork;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import shared.reader.SpecialCSVDataSetReader;
import shared.tester.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * Based on the XORTest test class, this class uses a standard FeedForwardNetwork
 * and various optimization problems.
 *
 * See numbered explanations for what each piece of the method does to address
 * the neural network optimization problem.
 *
 * @author Jesse Rosalia <https://github.com/theJenix>
 * @date 2013-03-05
 */
public class NeuralNetworkBreastCancerDataTest {

//    /**
//     * Tests out the perceptron with the classic xor test
//     * @param args ignored
//     */
//    public static void main(String[] args) throws Exception {
//        // Read data in
//        SpecialCSVDataSetReader csvDataSetReader = new SpecialCSVDataSetReader(new File("").getAbsolutePath() + "/src/opt/data/breast_cancer_wisconsin.data");
//        double[] labels = { 0, 1 };
//        Instance[] data = csvDataSetReader.read();
//        DataSet set = new DataSet(data);
//
//        // Define error measures
//        ErrorMeasure measure = new SumOfSquaresError();
//
//        // Second is RandomizedHillClimbing
//        System.out.println("1. Randomized Hill Climbing");
//        BackPropagationNetworkFactory feedForwardNeuralNetworkFactory = new BackPropagationNetworkFactory();
//        FeedForwardNetwork feedForwardNeuralNetwork = feedForwardNeuralNetworkFactory.createClassificationNetwork(new int[]{data.length, 10, 1});
//        NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(
//                set, feedForwardNeuralNetwork, measure);
//        OptimizationAlgorithm o = new RandomizedHillClimbing(nno);
//        FixedIterationTrainer fit = new FixedIterationTrainer(o, 500);
//        fit.train();
//        Instance opt = o.getOptimal();
//        feedForwardNeuralNetwork.setWeights(opt.getData());
//
//        TestMetric acc = new AccuracyTestMetric();
//        TestMetric cm  = new ConfusionMatrixTestMetric(labels);
//
//        Tester t = new NeuralNetworkTester(feedForwardNeuralNetwork, acc, cm);
//        t.test(data);
//
//        acc.printResults();
//        cm.printResults();
//
//        // Third is Simulated Annealing
//        System.out.println("2. Simulated Annealing");
//        o = new SimulatedAnnealing(10, 0.999, nno);
//        fit = new FixedIterationTrainer(o, 500);
//        fit.train();
//        opt = o.getOptimal();
//        feedForwardNeuralNetwork.setWeights(opt.getData());
//
//        acc = new AccuracyTestMetric();
//        cm  = new ConfusionMatrixTestMetric(labels);
//
//        t = new NeuralNetworkTester(feedForwardNeuralNetwork, acc, cm);
//        t.test(data);
//
//        acc.printResults();
//        cm.printResults();
//
//        // Finally Genetic Algorithm
//        System.out.println("3. Genetic Algorithm");
//        o = new StandardGeneticAlgorithm(10, 2, 2, nno);
//        fit = new FixedIterationTrainer(o, 500);
//        fit.train();
//        opt = o.getOptimal();
//        feedForwardNeuralNetwork.setWeights(opt.getData());
//
//        acc = new AccuracyTestMetric();
//        cm  = new ConfusionMatrixTestMetric(labels);
//
//        t = new NeuralNetworkTester(feedForwardNeuralNetwork, acc, cm);
//        t.test(data);
//
//        acc.printResults();
//        cm.printResults();
//    }
    private static int inputLayer = 9, hiddenLayer = 10, outputLayer = 1, trainingIterations = 1000, numInstances = 683;;
    private static Instance[] instances = initializeInstances();

    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for (Instance instance : instances) {
                networks[i].setInputValues(instance.getData());
                networks[i].run();

                predicted = Double.parseDouble(instance.getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for (Instance instance : instances) {
                network.setInputValues(instance.getData());
                network.run();

                Instance output = instance.getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {
        double[][][] attributes = new double[numInstances][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/data/breast_cancer_wisconsin.data")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[inputLayer]; // num attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < inputLayer; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
