package opt.test;

import java.util.Arrays;

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
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);



        System.out.println("Randomized Hill Climbing");
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.train();
        System.out.println(ef.value(rhc.getOptimal()));


        double temp = 1E12;
        double cooling = 0.95;
        System.out.println(String.format("Simulated Annealing (temp=%s, cooling=%s)", temp, cooling));
        SimulatedAnnealing sa = new SimulatedAnnealing(temp, cooling, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.train();
        System.out.println(ef.value(sa.getOptimal()));

        temp = 1E10;
        cooling = 0.95;
        System.out.println(String.format("Simulated Annealing (temp=%s, cooling=%s)", temp, cooling));
        SimulatedAnnealing sa_10 = new SimulatedAnnealing(temp, cooling, hcp);
        fit = new FixedIterationTrainer(sa_10, 200000);
        fit.train();
        System.out.println(ef.value(sa_10.getOptimal()));

        temp = 1E14;
        cooling = 0.95;
        System.out.println(String.format("Simulated Annealing (temp=%s, cooling=%s)", temp, cooling));
        SimulatedAnnealing sa_14 = new SimulatedAnnealing(temp, cooling, hcp);
        fit = new FixedIterationTrainer(sa_14, 200000);
        fit.train();
        System.out.println(ef.value(sa_14.getOptimal()));

        temp = 1E12;
        cooling = .90;
        System.out.println(String.format("Simulated Annealing (temp=%s, cooling=%s)", temp, cooling));
        SimulatedAnnealing sa90 = new SimulatedAnnealing(temp, cooling, hcp);
        fit = new FixedIterationTrainer(sa90, 200000);
        fit.train();
        System.out.println(ef.value(sa90.getOptimal()));

        temp = 1E12;
        cooling = .99;
        System.out.println(String.format("Simulated Annealing (temp=%s, cooling=%s)", temp, cooling));
        SimulatedAnnealing sa99 = new SimulatedAnnealing(temp, cooling, hcp);
        fit = new FixedIterationTrainer(sa99, 200000);
        fit.train();
        System.out.println(ef.value(sa99.getOptimal()));



        int populationSize = 200;
        int toMate = 150;
        int toMutate = 20;
        System.out.println(String.format("Genetic Algorithm (populationSize=%s, toMate=%s, toMutate=%s)", populationSize, toMate, toMutate));
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        System.out.println(ef.value(ga.getOptimal()));

        populationSize = 150;
        toMate = 150;
        toMutate = 20;
        System.out.println(String.format("Genetic Algorithm (populationSize=%s, toMate=%s, toMutate=%s)", populationSize, toMate, toMutate));
        StandardGeneticAlgorithm ga_100_pop = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, gap);
        fit = new FixedIterationTrainer(ga_100_pop, 1000);
        fit.train();
        System.out.println(ef.value(ga_100_pop.getOptimal()));

        populationSize = 300;
        toMate = 150;
        toMutate = 20;
        System.out.println(String.format("Genetic Algorithm (populationSize=%s, toMate=%s, toMutate=%s)", populationSize, toMate, toMutate));
        StandardGeneticAlgorithm ga_300_pop = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, gap);
        fit = new FixedIterationTrainer(ga_300_pop, 1000);
        fit.train();
        System.out.println(ef.value(ga_300_pop.getOptimal()));

        populationSize = 200;
        toMate = 100;
        toMutate = 20;
        System.out.println(String.format("Genetic Algorithm (populationSize=%s, toMate=%s, toMutate=%s)", populationSize, toMate, toMutate));
        StandardGeneticAlgorithm ga_100_mate = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, gap);
        fit = new FixedIterationTrainer(ga_100_mate, 1000);
        fit.train();
        System.out.println(ef.value(ga_100_mate.getOptimal()));

        populationSize = 200;
        toMate = 50;
        toMutate = 20;
        System.out.println(String.format("Genetic Algorithm (populationSize=%s, toMate=%s, toMutate=%s)", populationSize, toMate, toMutate));
        StandardGeneticAlgorithm ga_50_mate = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, gap);
        fit = new FixedIterationTrainer(ga_50_mate, 1000);
        fit.train();
        System.out.println(ef.value(ga_50_mate.getOptimal()));

        populationSize = 200;
        toMate = 200;
        toMutate = 20;
        System.out.println(String.format("Genetic Algorithm (populationSize=%s, toMate=%s, toMutate=%s)", populationSize, toMate, toMutate));
        StandardGeneticAlgorithm ga_200_mate = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, gap);
        fit = new FixedIterationTrainer(ga_200_mate, 1000);
        fit.train();
        System.out.println(ef.value(ga_200_mate.getOptimal()));

        populationSize = 200;
        toMate = 150;
        toMutate = 50;
        System.out.println(String.format("Genetic Algorithm (populationSize=%s, toMate=%s, toMutate=%s)", populationSize, toMate, toMutate));
        StandardGeneticAlgorithm ga_50_mutate = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, gap);
        fit = new FixedIterationTrainer(ga_50_mutate, 1000);
        fit.train();
        System.out.println(ef.value(ga_50_mutate.getOptimal()));

        populationSize = 200;
        toMate = 150;
        toMutate = 0;
        System.out.println(String.format("Genetic Algorithm (populationSize=%s, toMate=%s, toMutate=%s)", populationSize, toMate, toMutate));
        StandardGeneticAlgorithm ga_0_mutate = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, gap);
        fit = new FixedIterationTrainer(ga_0_mutate, 1000);
        fit.train();
        System.out.println(ef.value(ga_0_mutate.getOptimal()));
    }
}
