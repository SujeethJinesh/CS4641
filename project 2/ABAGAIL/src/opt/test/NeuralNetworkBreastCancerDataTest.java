package opt.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
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
    private static int inputLayer = 9, hiddenLayer = 4, outputLayer = 1, trainingIterations = 200, numTrainingInstances = 545, numTestingInstances = 138;
    private static Instance[] trainingInstances;
    private static Instance[] testingInstances;

    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) throws IOException {
        initializeInstances();
        DataSet trainingSet = new DataSet(trainingInstances);

        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(trainingSet, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correctBenign = 0, incorrectBenign = 0, correctMalignant = 0, incorrectMalignant = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            ArrayList<Double> actuals = new ArrayList<>();
            ArrayList<Double> predicteds = new ArrayList<>();
            for (Instance instance : testingInstances) {
                networks[i].setInputValues(instance.getData());
                networks[i].run();

                actual = Double.parseDouble(instance.getLabel().toString());
                predicted = Double.parseDouble(networks[i].getOutputValues().toString());

                actuals.add(actual);
                predicteds.add((double) Math.round(predicted));

                if (actual == 0) {
                    if (Math.round(predicted) == 0) {
                        correctBenign++;
                    } else {
                        incorrectBenign++;
                    }
                } else {
                    if (Math.round(predicted) == 1) {
                        correctMalignant++;
                    } else {
                        incorrectMalignant++;
                    }
                }
            }

            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);
            double totalCorrect = (correctBenign + correctMalignant);
            double totalIncorrect =  (incorrectBenign + incorrectMalignant);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + totalCorrect + " instances." +
                    "\nIncorrectly classified " + totalIncorrect + " instances.\nPercent correctly classified: "
                    + df.format(totalCorrect/(totalCorrect+totalIncorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            results += "\t\t\tPred. Benign\tPred. Malignant\n";
            results += "Act. Benign\t\t" + correctBenign + "\t\t" + incorrectBenign + "\n";
            results += "Act. Malignant\t\t" + incorrectMalignant + "\t\t" + correctMalignant + "\n";

            FileWriter writer = new FileWriter(oaNames[i] + "_actuals.csv");
            for(double doub: actuals) {
                writer.write(doub + ",");
            }
            writer.close();

            writer = new FileWriter(oaNames[i] + "_predicteds.csv");
            for(double doub: predicteds) {
                writer.write(doub + ",");
            }
            writer.close();
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) throws IOException {
        System.out.println("\nError results for " + oaName + "\n---------------------------");
        ArrayList<Double> errors = new ArrayList<>();
        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for (Instance instance : trainingInstances) {
                network.setInputValues(instance.getData());
                network.run();

                Instance output = instance.getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }
            errors.add(error);
            System.out.println(df.format(error));
        }

        FileWriter writer = new FileWriter(oaName + "_errors.csv");
        for(double doub: errors) {
            writer.write(doub + ",");
        }
        writer.close();
    }

    private static void initializeInstances() {
        double[][][] trainingAttributes = new double[numTrainingInstances][][];
        double[][][] testingAttributes = new double[numTestingInstances][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/data/breast_cancer_wisconsin.data")));

            for(int i = 0; i < trainingAttributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                trainingAttributes[i] = new double[2][];
                trainingAttributes[i][0] = new double[inputLayer]; // num attributes
                trainingAttributes[i][1] = new double[1];

                for(int j = 0; j < inputLayer; j++)
                    trainingAttributes[i][0][j] = Double.parseDouble(scan.next());

                trainingAttributes[i][1][0] = Double.parseDouble(scan.next());
            }

            for(int i = trainingAttributes.length; i < testingAttributes.length + trainingAttributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");
                int index = i - trainingAttributes.length;
                testingAttributes[index] = new double[2][];
                testingAttributes[index][0] = new double[inputLayer]; // num attributes
                testingAttributes[index][1] = new double[1];

                for(int j = 0; j < inputLayer; j++)
                    testingAttributes[index][0][j] = Double.parseDouble(scan.next());

                testingAttributes[index][1][0] = Double.parseDouble(scan.next());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        trainingInstances = new Instance[trainingAttributes.length];
        testingInstances = new Instance[testingAttributes.length];

        for(int i = 0; i < trainingInstances.length; i++) {
            trainingInstances[i] = new Instance(trainingAttributes[i][0]);
            trainingInstances[i].setLabel(new Instance(trainingAttributes[i][1][0]));
        }

        for(int i = 0; i < testingInstances.length; i++) {
            testingInstances[i] = new Instance(testingAttributes[i][0]);
            testingInstances[i].setLabel(new Instance(testingAttributes[i][1][0]));
        }
    }
}
