package edu.uic.cs.purposeful.mpg.target.linear_chain.f1;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.util.BitSet;
import java.util.List;

import org.apache.commons.lang3.tuple.Triple;
import org.junit.Test;
import org.junit.Ignore;

import com.google.common.collect.Iterables;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.common.Misc;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.common.ValuePrecision;
import edu.uic.cs.purposeful.mpg.learning.MaximizerPredictor.Prediction;
import edu.uic.cs.purposeful.mpg.learning.linear_chain.LinearChainDataSet;
import edu.uic.cs.purposeful.mpg.learning.linear_chain.MPGLinearChainClassifer;
import edu.uic.cs.purposeful.mpg.target.linear_chain.LinearChain;

public class TestLinearChainF1Classifier {

  @Ignore
  @Test
  public void testLearnAndPredictOnSameData_binarized() throws Exception {
    int[] targetTags = new int[] {1, 2, 5};

    for (int targetTag : targetTags) {
      MPGLinearChainClassifer classifier =
          new MPGLinearChainClassifer(LinearChainF1.class, targetTag);
      File file = new File(TestLinearChainF1Classifier.class
          .getResource("TestLinearChainF1Classifier.train").toURI());
      LinearChainDataSet dataSet = LinearChainDataSet.loadFromFile(file);
      dataSet = LinearChainDataSet.binarize(dataSet, targetTag);

      double[] thetas = classifier.learn(dataSet, Regularization.l2(0.001));

      List<Prediction<LinearChain>> predictions = classifier.predict(dataSet);
      Prediction<LinearChain> prediction = Iterables.getOnlyElement(predictions);
      Triple<Double, Double, Double> precisionRecallF1 = evaluate(predictions, targetTag);

      System.out.println("targetTag:\t" + targetTag);
      System.out.println("Thetas:\t" + Misc.toDisplay(thetas));
      System.out.println("Golden:\t" + prediction.getGoldenPermutation());
      System.out.println("Prediction:\t" + prediction.getPredictionPermutation());
      System.out.println("Precision:\t" + precisionRecallF1.getLeft());
      System.out.println("Recall:\t" + precisionRecallF1.getMiddle());
      System.out.println("F1:\t" + precisionRecallF1.getRight());
      assertEquals(prediction.getScore(), precisionRecallF1.getRight(),
          ValuePrecision.POINT_8_ZEROS_ONE.getValuePrecision());
      System.out.println("Probability:\t" + prediction.getProbability());

      assertEquals(prediction.getGoldenPermutation().getExistenceSequence(targetTag),
          prediction.getPredictionPermutation().getExistenceSequence(targetTag));
      System.out.println();
    }
  }
  //@Ignore
  @Test
  public void testLearnAndPredictOnSameData() throws Exception {
    int[] targetTags = new int[] {1, 2, 5};

    for (int targetTag : targetTags) {
      MPGLinearChainClassifer classifier =
          new MPGLinearChainClassifer(LinearChainF1.class, targetTag);
      File file = new File(TestLinearChainF1Classifier.class
          .getResource("TestLinearChainF1Classifier.train").toURI());
      LinearChainDataSet dataSet = LinearChainDataSet.loadFromFile(file);

      double[] thetas = classifier.learn(dataSet, Regularization.l2(0.001));

      List<Prediction<LinearChain>> predictions = classifier.predict(dataSet);
      Prediction<LinearChain> prediction = Iterables.getOnlyElement(predictions);
      Triple<Double, Double, Double> precisionRecallF1 = evaluate(predictions, targetTag);

      System.out.println("targetTag:\t" + targetTag);
      System.out.println("Thetas:\t" + Misc.toDisplay(thetas));
      System.out.println("Golden:\t" + prediction.getGoldenPermutation());
      System.out.println("Prediction:\t" + prediction.getPredictionPermutation());
      System.out.println("Precision:\t" + precisionRecallF1.getLeft());
      System.out.println("Recall:\t" + precisionRecallF1.getMiddle());
      System.out.println("F1:\t" + precisionRecallF1.getRight());
      assertEquals(prediction.getScore(), precisionRecallF1.getRight(),
          ValuePrecision.POINT_8_ZEROS_ONE.getValuePrecision());
      System.out.println("Probability:\t" + prediction.getProbability());

      assertEquals(prediction.getGoldenPermutation().getExistenceSequence(targetTag),
          prediction.getPredictionPermutation().getExistenceSequence(targetTag));
      System.out.println();
    }
  }

  private static Triple<Double, Double, Double> evaluate(List<Prediction<LinearChain>> predictions,
      int targetTag) {
    int[][] confusionMatrix = new int[2][2];

    for (Prediction<LinearChain> prediction : predictions) {
      LinearChain goldenLinearChain = prediction.getGoldenPermutation();
      LinearChain predictedLinearChain = prediction.getPredictionPermutation();
      Assert.isTrue(goldenLinearChain.getLength() == predictedLinearChain.getLength());

      BitSet goldenBinarySequence = goldenLinearChain.getExistenceSequence(targetTag);
      BitSet predictedBinarySequence = predictedLinearChain.getExistenceSequence(targetTag);
      for (int index = 0; index < goldenLinearChain.getLength(); index++) {
        int golden = goldenBinarySequence.get(index) ? 1 : 0;
        int predicted = predictedBinarySequence.get(index) ? 1 : 0;
        confusionMatrix[golden][predicted] += 1;
      }
    }

    double precision = precision(confusionMatrix);
    double recall = recall(confusionMatrix);
    double f1 = f1(precision, recall);
    return Triple.of(precision, recall, f1);
  }

  private static double f1(double precision, double recall) {
    if ((precision + recall) == 0) {
      return 0;
    }
    return 2 * precision * recall / (precision + recall);
  }

  private static double precision(int[][] confusionMatrix) {
    double correct = 0;
    double total = 0;
    for (int i = 0; i < 2; i++) {
      if (i == 1) {
        correct += confusionMatrix[i][1];
      }
      total += confusionMatrix[i][1];
    }
    if (total == 0) {
      return 0;
    }
    return correct / total;
  }

  private static double recall(int[][] confusionMatrix) {
    double correct = 0;
    double total = 0;
    for (int j = 0; j < 2; j++) {
      if (j == 1) {
        correct += confusionMatrix[1][j];
      }
      total += confusionMatrix[1][j];
    }
    if (total == 0) {
      return 0;
    }
    return correct / total;
  }
}
