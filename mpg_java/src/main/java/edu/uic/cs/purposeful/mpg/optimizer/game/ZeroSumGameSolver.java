package edu.uic.cs.purposeful.mpg.optimizer.game;

import edu.uic.cs.purposeful.mpg.optimizer.numerical.IterationCallback;

import java.util.LinkedHashSet;

public interface ZeroSumGameSolver<Permutation> {
  boolean solve(double[] thetas);

  boolean solve(double[] thetas, Permutation goldPermutation);

  boolean solve(double[] thetas, Permutation goldPermutation, IterationCallback callback);

  double getMaximizerValue();

  double getMinimizerValue();

  double[] getMaximizerProbabilities();

  double[] getMinimizerProbabilities();

  LinkedHashSet<Permutation> getMaximizerPermutations();

  LinkedHashSet<Permutation> getMinimizerPermutations();
}
