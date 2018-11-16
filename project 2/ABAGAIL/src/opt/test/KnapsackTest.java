package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 50;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum weight for the knapsack */
    private static final double MAX_KNAPSACK_WEIGHT =
         MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[NUM_ITEMS];
        double[] weights = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);

        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.train();
        System.out.println(ef.value(rhc.getOptimal()));




        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.train();
        System.out.println(ef.value(sa.getOptimal()));

        SimulatedAnnealing sa_10 = new SimulatedAnnealing(1E10, .95, hcp);
        fit = new FixedIterationTrainer(sa_10, 200000);
        fit.train();
        System.out.println(ef.value(sa_10.getOptimal()));

        SimulatedAnnealing sa_14 = new SimulatedAnnealing(1E14, .95, hcp);
        fit = new FixedIterationTrainer(sa_14, 200000);
        fit.train();
        System.out.println(ef.value(sa_14.getOptimal()));

        SimulatedAnnealing sa90 = new SimulatedAnnealing(1E12, .90, hcp);
        fit = new FixedIterationTrainer(sa90, 200000);
        fit.train();
        System.out.println(ef.value(sa90.getOptimal()));

        SimulatedAnnealing sa99 = new SimulatedAnnealing(1E12, .99, hcp);
        fit = new FixedIterationTrainer(sa99, 200000);
        fit.train();
        System.out.println(ef.value(sa99.getOptimal()));




        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        System.out.println(ef.value(ga.getOptimal()));

        StandardGeneticAlgorithm ga_100_pop = new StandardGeneticAlgorithm(150, 150, 20, gap);
        fit = new FixedIterationTrainer(ga_100_pop, 1000);
        fit.train();
        System.out.println(ef.value(ga_100_pop.getOptimal()));

        StandardGeneticAlgorithm ga_300_pop = new StandardGeneticAlgorithm(300, 150, 20, gap);
        fit = new FixedIterationTrainer(ga_300_pop, 1000);
        fit.train();
        System.out.println(ef.value(ga_300_pop.getOptimal()));

        StandardGeneticAlgorithm ga_100_mate = new StandardGeneticAlgorithm(200, 100, 20, gap);
        fit = new FixedIterationTrainer(ga_100_mate, 1000);
        fit.train();
        System.out.println(ef.value(ga_100_mate.getOptimal()));

        StandardGeneticAlgorithm ga_200_mate = new StandardGeneticAlgorithm(200, 200, 20, gap);
        fit = new FixedIterationTrainer(ga_200_mate, 1000);
        fit.train();
        System.out.println(ef.value(ga_200_mate.getOptimal()));

        StandardGeneticAlgorithm ga_50_mutate = new StandardGeneticAlgorithm(200, 150, 50, gap);
        fit = new FixedIterationTrainer(ga_50_mutate, 1000);
        fit.train();
        System.out.println(ef.value(ga_50_mutate.getOptimal()));

        StandardGeneticAlgorithm ga_0_mutate = new StandardGeneticAlgorithm(200, 150, 0, gap);
        fit = new FixedIterationTrainer(ga_0_mutate, 1000);
        fit.train();
        System.out.println(ef.value(ga_0_mutate.getOptimal()));
    }

}
