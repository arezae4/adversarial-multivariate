package edu.uic.cs.purposeful.mpg.learning.linear_chain;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.common.reflect.ClassFactory;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.Misc;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.learning.MaximizerPredictor;
import edu.uic.cs.purposeful.mpg.learning.linear_chain.LinearChainDataSet.FeatureType;
import edu.uic.cs.purposeful.mpg.learning.linear_chain.LinearChainDataSet.LinearChainDataSetInstance;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.IterationCallback;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.NumericalOptimizer;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.objective.impl.MinimizationObjectiveFunctionImpl;
import edu.uic.cs.purposeful.mpg.target.OptimizationTarget;
import edu.uic.cs.purposeful.mpg.target.linear_chain.LinearChain;

public class MPGLinearChainClassifer
    extends MaximizerPredictor<LinearChain, Pair<Integer, LinearChainDataSetInstance>> {
  private static final Logger LOGGER = Logger.getLogger(MPGLinearChainClassifer.class);

  private static final Random RANDOM = new Random(2147483647L);
  private static final double SMALL_CONST = 1e-6;

  private final Integer targetTag;
  private double[] thetas;

  public MPGLinearChainClassifer(
      Class<? extends OptimizationTarget<LinearChain, Pair<Integer, LinearChainDataSetInstance>>> optimizationTargetClass,
      Integer targetTag) {
    super(optimizationTargetClass);
    this.targetTag = targetTag;
  }

  public double[] learn(File trainingDataFile, Regularization regularization) {
    LinearChainDataSet trainingDataset = LinearChainDataSet.loadFromFile(trainingDataFile);
    return learn(trainingDataset, regularization);
  }

  public double[] learn(LinearChainDataSet trainingDataset, Regularization regularization) {
    return learn(trainingDataset, regularization, null);
  }

  public double[] learn(LinearChainDataSet trainingDataset, Regularization regularization,
      IterationCallback iterationCallback) {
    thetas = initializeThetas(trainingDataset, regularization);
    if (LOGGER.isInfoEnabled()) {
      LOGGER.info("Initialized thetas: " + Misc.toDisplay(thetas));
    }

    NumericalOptimizer numericalOptimizer =
        ClassFactory.getInstance(MPGConfig.NUMERICAL_OPTIMIZER_IMPLEMENTATION);
    LOGGER.warn("Using numerical optimizer implementation: " + numericalOptimizer.getClass());
    numericalOptimizer.setMinimizationObjectiveFunction(new MinimizationObjectiveFunctionImpl<>(
        optimizationTargetClass, bindInstancesWithTags(trainingDataset.getInstances(), targetTag)));
    boolean converged = numericalOptimizer.optimize(thetas, regularization, iterationCallback);
    LOGGER.warn("Finish optimization using numerical optimizer, converged=" + converged);

    return thetas;
  }

  private List<Pair<Integer, LinearChainDataSetInstance>> bindInstancesWithTags(
      List<LinearChainDataSetInstance> instances, Integer targetTag) {
    List<Pair<Integer, LinearChainDataSetInstance>> results = new ArrayList<>(instances.size());
    for (LinearChainDataSetInstance instance : instances) {
      results.add(Pair.of(targetTag, instance));
    }
    return results;
  }

  private double[] initializeThetas(LinearChainDataSet trainingDataset,
      Regularization regularization) {
    int sizeOfUnigramFeatureVectors =
        trainingDataset.getNumOfTags() * trainingDataset.getNumOfUnigramFeatures();
    int sizeOfBigramFeatureVectors =
        trainingDataset.getNumOfTagPairs() * trainingDataset.getNumOfBigramFeatures();

    double[] thetas = new double[sizeOfUnigramFeatureVectors + sizeOfBigramFeatureVectors];

    if (MPGConfig.LEARN_INITIAL_THETAS) {
      LogisticRegressionLinearChain logisticRegression =
          new LogisticRegressionLinearChain(FeatureType.U);
      double[] unigramWeights = logisticRegression.learnWeights(trainingDataset, regularization);
      Assert.isTrue(unigramWeights.length == sizeOfUnigramFeatureVectors);
      System.arraycopy(unigramWeights, 0, thetas, 0, unigramWeights.length);

      for (int i = unigramWeights.length; i < thetas.length; i++) {
        thetas[i] = RANDOM.nextDouble() + SMALL_CONST;
      }
    }
    return thetas;
  }

  public List<Prediction<LinearChain>> predict(File testDataFile) {
    Assert.notNull(thetas, "Learn or load the model first.");
    return predict(testDataFile, thetas);
  }

  public List<Prediction<LinearChain>> predict(LinearChainDataSet testDataset) {
    return predict(testDataset, thetas);
  }

  public List<Prediction<LinearChain>> predict(File testDataFile, double[] thetas) {
    LinearChainDataSet testDataset = LinearChainDataSet.loadFromFile(testDataFile);
    return predict(testDataset, thetas);
  }

  public List<Prediction<LinearChain>> predict(LinearChainDataSet testDataset, double[] thetas) {
    int numOfTags = testDataset.getNumOfTags();
    int numOfTagPairs = testDataset.getNumOfTagPairs();
    int numOfUnigramFeatures = testDataset.getNumOfUnigramFeatures();
    int numOfBigramFeatures = testDataset.getNumOfBigramFeatures();

    int length = numOfTags * numOfUnigramFeatures + numOfTagPairs * numOfBigramFeatures;
    Assert.isTrue(length == thetas.length,
        String.format(
            "numOfTags[%d] * numOfUnigramFeatures[%d] + numOfTagPairs[%d] * numOfBigramFeatures[%d] = %d != thetas.length[%d]",
            numOfTags, numOfUnigramFeatures, numOfTagPairs, numOfBigramFeatures, length,
            thetas.length));

    return predict(bindInstancesWithTags(testDataset.getInstances(), targetTag), thetas);
  }

  public void writeModel(File modelFile) {
    writeModelToFile(modelFile, thetas);
  }

  public void loadModel(File modelFile) {
    thetas = loadModelFromFile(modelFile);
  }
}
