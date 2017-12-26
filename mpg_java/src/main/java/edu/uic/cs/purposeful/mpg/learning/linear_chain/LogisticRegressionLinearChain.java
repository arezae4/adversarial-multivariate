package edu.uic.cs.purposeful.mpg.learning.linear_chain;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.math3.util.MathUtils;
import org.apache.log4j.Logger;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.Norm;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.learning.LogisticRegressionHelper;
import edu.uic.cs.purposeful.mpg.learning.linear_chain.LinearChainDataSet.FeatureType;
import edu.uic.cs.purposeful.mpg.learning.linear_chain.LinearChainDataSet.LinearChainDataSetInstance;
import gnu.trove.list.array.TDoubleArrayList;
import no.uib.cipr.matrix.MatrixEntry;

public class LogisticRegressionLinearChain
    implements LogisticRegressionHelper<int[], LinearChainDataSet> {
  private static final Logger LOGGER = Logger.getLogger(LogisticRegressionLinearChain.class);

  private final FeatureType featureType;

  public LogisticRegressionLinearChain(FeatureType featureTypeToUse) {
    this.featureType = featureTypeToUse;
  }

  private Problem convertToNonLinearChain(LinearChainDataSet originalDataSet) {
    int numOfFeatures = 0;
    if (featureType == null) {
      numOfFeatures =
          originalDataSet.getNumOfUnigramFeatures() + originalDataSet.getNumOfBigramFeatures();
    } else if (featureType == FeatureType.U) {
      numOfFeatures = originalDataSet.getNumOfUnigramFeatures();
    } else if (featureType == FeatureType.B) {
      numOfFeatures = originalDataSet.getNumOfBigramFeatures();
    } else {
      Assert.isTrue(false);
    }

    List<Feature[]> featuress = new ArrayList<>();
    TDoubleArrayList goldenTags = new TDoubleArrayList();

    for (LinearChainDataSetInstance instance : originalDataSet.getInstances()) {
      TreeMap<Integer, ArrayList<Feature>> featuresByPosition = new TreeMap<>();
      for (int rowIndex = 0; rowIndex < instance.getGoldenTags().length; rowIndex++) {
        featuresByPosition.put(rowIndex, new ArrayList<>());
        goldenTags.add(instance.getGoldenTags()[rowIndex]);
      }

      if (featureType == FeatureType.U || featureType == null) {
        for (MatrixEntry entry : instance.getUnigramFeatureMatrix()) {
          int featureIndex = entry.column() + 1; // to 1-based
          double featureValue = entry.get();
          ArrayList<Feature> features = featuresByPosition.get(entry.row());
          features.add(new FeatureNode(featureIndex, featureValue));
        }
      }

      if (featureType == FeatureType.B || featureType == null) {
        int featureBaseIndex =
            (featureType == null) ? originalDataSet.getNumOfUnigramFeatures() : 0;
        for (MatrixEntry entry : instance.getBigramFeatureMatrix()) {
          int featureIndex = featureBaseIndex + entry.column() + 1; // to 1-based
          double featureValue = entry.get();
          ArrayList<Feature> features = featuresByPosition.get(entry.row());
          features.add(new FeatureNode(featureIndex, featureValue));
        }
      }

      for (ArrayList<Feature> features : featuresByPosition.values()) {
        featuress.add(features.toArray(new Feature[features.size()]));
      }
    }

    Problem problem = new Problem();
    problem.bias = -1; // bias feature has already been handled by original data set
    problem.l = featuress.size();
    problem.n = numOfFeatures;
    problem.y = goldenTags.toArray();
    problem.x = featuress.toArray(new Feature[featuress.size()][]);
    return problem;
  }

  @Override
  public Model learnModel(LinearChainDataSet dataSet, Regularization regularization) {
    Problem problem = convertToNonLinearChain(dataSet);

    double regularizationParameter = regularization.getParameter();
    if (MathUtils.equals(regularizationParameter, 0)) {
      LOGGER.warn(
          "[regularizationParameter==0], in such case, Logistic Regression (LibLinear) would learn all zeros as its weights.");
      regularizationParameter = Double.MIN_VALUE; // use very small number to approximate zero
    } else {
      // LR's regularization is the reciprocal of the regularization in MPG
      regularizationParameter = 1.0 / regularizationParameter;
    }

    Parameter param = null;
    if (regularization.getNorm() == Norm.L1) {
      param = new Parameter(SolverType.L1R_LR, regularizationParameter,
          MPGConfig.LOGISTIC_REGRESSION_STOPPING_CRITERION);
    } else if (regularization.getNorm() == Norm.L2) {
      param = new Parameter(SolverType.L2R_LR, regularizationParameter,
          MPGConfig.LOGISTIC_REGRESSION_STOPPING_CRITERION);
    } else {
      Assert.isTrue(false);
    }

    Linear.disableDebugOutput();
    return Linear.train(problem, param);
  }

  @Override
  public double[] learnWeights(LinearChainDataSet dataSet, Regularization regularization) {
    Model model = learnModel(dataSet, regularization);
    double[] weights = model.getFeatureWeights();
    int[] tags = model.getLabels();

    Map<Integer, Integer> indicesByTag = dataSet.getIndicesByTag();
    Assert.isTrue(tags.length == dataSet.getNumOfTags());
    double[] resultWeights = null;
    if (tags.length == 2) { // learned weights are for tags[0]
      resultWeights = new double[weights.length * 2];
      int index = indicesByTag.get(tags[0]);
      Assert.isTrue(index == 0 || index == 1);
      int learnedWeightStartIndex = index * weights.length;
      int negativeLearnedWeightStartIndex = (1 - index) * weights.length;
      for (int weightIndex = 0; weightIndex < weights.length; weightIndex++) {
        resultWeights[learnedWeightStartIndex + weightIndex] = weights[weightIndex];
        resultWeights[negativeLearnedWeightStartIndex + weightIndex] = -weights[weightIndex];
      }
    } else {
      resultWeights = new double[weights.length];
      int numOfFeatures = weights.length / tags.length;

      int index = 0;
      for (int featureIndex = 0; featureIndex < numOfFeatures; featureIndex++) {
        for (int tag : tags) {
          int tagIndex = indicesByTag.get(tag);
          resultWeights[tagIndex * numOfFeatures + featureIndex] = weights[index++];
        }
      }
    }
    return resultWeights;
  }

  @Override
  public int[] predict(LinearChainDataSet dataSet, Model model) {
    Linear.disableDebugOutput();
    Problem problem = convertToNonLinearChain(dataSet);

    int[] tagSequence = new int[problem.l];
    for (int positionIndex = 0; positionIndex < problem.l; positionIndex++) {
      tagSequence[positionIndex] = (int) Linear.predict(model, problem.x[positionIndex]);
    }

    return tagSequence;
  }
}
