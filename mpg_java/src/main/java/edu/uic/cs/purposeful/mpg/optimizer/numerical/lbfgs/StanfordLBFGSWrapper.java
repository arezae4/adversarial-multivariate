package edu.uic.cs.purposeful.mpg.optimizer.numerical.lbfgs;

import org.apache.log4j.Logger;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.FeatureWiseRegularization;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.IterationCallback;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.NumericalOptimizer;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.lbfgs.stanford.StanfordCoreNLPQNMinimizerLite;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.objective.MinimizationObjectiveFunction;

public class StanfordLBFGSWrapper implements NumericalOptimizer {
  private static final Logger LOGGER = Logger.getLogger(StanfordLBFGSWrapper.class);

  private static final int NUMBER_OF_PREVIOUS_ESTIMATIONS = 15;

  private MinimizationObjectiveFunction objectiveFunction;

  @Override
  public void setMinimizationObjectiveFunction(MinimizationObjectiveFunction objectiveFunction) {
    this.objectiveFunction = objectiveFunction;
  }

  @Override
  public boolean optimize(double[] thetas, Regularization regularization) {
    return optimize(thetas, regularization, null);
  }

  @Override
  public boolean optimize(double[] thetas, Regularization regularization,
      IterationCallback iterationCallback) {
    objectiveFunction.setRegularization(regularization);
    return optimize(thetas, iterationCallback);
  }

  @Override
  public boolean optimize(double[] thetas, FeatureWiseRegularization featureWiseRegularization) {
    return optimize(thetas, featureWiseRegularization, null);
  }

  @Override
  public boolean optimize(double[] thetas, FeatureWiseRegularization featureWiseRegularization,
      IterationCallback iterationCallback) {
    objectiveFunction.setRegularization(featureWiseRegularization);
    return optimize(thetas, iterationCallback);
  }

  private boolean optimize(double[] thetas, IterationCallback iterationCallback) {
    StanfordCoreNLPQNMinimizerLite lbfgs =
        new StanfordCoreNLPQNMinimizerLite(NUMBER_OF_PREVIOUS_ESTIMATIONS);
    lbfgs.terminateOnMaxItr(MPGConfig.LBFGS_MAX_NUMBER_OF_ITERATIONS);
    lbfgs.shutUp();
    double[] optimalThetas = lbfgs.minimize(objectiveFunction, thetas, iterationCallback);

    Assert.isTrue(optimalThetas.length == thetas.length);
    System.arraycopy(optimalThetas, 0, thetas, 0, optimalThetas.length);

    LOGGER.info(lbfgs.getState());
    return lbfgs.wasSuccessful();
  }

}
