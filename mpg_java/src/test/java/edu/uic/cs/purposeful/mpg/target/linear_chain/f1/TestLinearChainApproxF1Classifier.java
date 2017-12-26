package edu.uic.cs.purposeful.mpg.target.linear_chain.f1;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.base.Stopwatch;
import edu.uic.cs.purposeful.mpg.optimizer.numerical.IterationCallback;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Triple;
import org.junit.Rule;
import org.junit.Test;
import org.junit.Ignore;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.common.Misc;
import edu.uic.cs.purposeful.mpg.common.Regularization;
import edu.uic.cs.purposeful.mpg.learning.MaximizerPredictor.Prediction;
import edu.uic.cs.purposeful.mpg.learning.linear_chain.LinearChainDataSet;
import edu.uic.cs.purposeful.mpg.learning.linear_chain.MPGLinearChainClassifer;
import edu.uic.cs.purposeful.mpg.target.linear_chain.LinearChain;
import org.junit.rules.TestName;

public class TestLinearChainApproxF1Classifier {


  @Rule public TestName testName = new TestName();

  //@Ignore
  @Test
  public void testLearnAndPredict_binarized() throws Exception {

    System.out.println("TESTING: " + testName.getMethodName());
    int[] targetTags = new int[] {1, 2};
    File file = new File(TestLinearChainApproxF1Classifier.class
          .getResource("TestLinearChainF1.300.train").toURI());
    File testfile = new File(TestLinearChainApproxF1Classifier.class
          .getResource("TestLinearChainF1.300.test").toURI());

    LinearChainDataSet dataSet = LinearChainDataSet.loadFromFile(file);
    LinearChainDataSet testDataSet = LinearChainDataSet.loadFromFile(testfile);

    for (int targetTag : targetTags) {
      MPGLinearChainClassifer classifierCS =
          new MPGLinearChainClassifer(LinearChainApproxF1.class, targetTag);
      MPGLinearChainClassifer classifier2 =
          new MPGLinearChainClassifer(LinearChainF1.class, targetTag);

      LinearChainDataSet binDataSet = LinearChainDataSet.binarize(dataSet, targetTag);
      LinearChainDataSet binTestDataSet = LinearChainDataSet.binarize(testDataSet, targetTag);

      Stopwatch watchCS = Stopwatch.createStarted();

      double[] thetasCS = classifierCS.learn(binDataSet, Regularization.l2(0.001));
      watchCS.stop();

      List<Prediction<LinearChain>> predictionsCS = classifier2.predict(binTestDataSet,thetasCS);
      Triple<Double, Double, Double> precisionRecallF1CS = evaluate(predictionsCS, targetTag,binTestDataSet.getIndicesByTag());

      System.out.println("Trained Classifier:\t CostSensitiveApproximatedF1 (~)");
      System.out.println("targetTag:\t" + targetTag);
      System.out.println("Precision:\t" + precisionRecallF1CS.getLeft() );
      System.out.println("Recall:\t" + precisionRecallF1CS.getMiddle() );
      System.out.println("F1:\t" + precisionRecallF1CS.getRight() );
      System.out.println("time (~) (*):\t" + watchCS.elapsed(TimeUnit.SECONDS) );
      System.out.println();
    }
  }

  @Test
  public void testLearnAndPredict() throws Exception {

    System.out.println("TESTING: " + testName.getMethodName());
    File file = new File(TestLinearChainApproxF1Classifier.class
          .getResource("TestLinearChainF1.100.train").toURI());
    File testfile = new File(TestLinearChainApproxF1Classifier.class
          .getResource("TestLinearChainF1.100.test").toURI());

    LinearChainDataSet dataSet = LinearChainDataSet.loadFromFile(file);
    LinearChainDataSet testDataSet = LinearChainDataSet.loadFromFile(testfile);

    int[] targetTags = new int[] {1,2,4};

    for (int targetTag : targetTags) {
      MPGLinearChainClassifer classifierApprox =
          new MPGLinearChainClassifer(LinearChainApproxF1.class, targetTag);
      MPGLinearChainClassifer classifierExact =
          new MPGLinearChainClassifer(LinearChainApproxF1.class, targetTag);


      System.out.println("TRAINING a model with approximated F1 score");
      Stopwatch watchCS = Stopwatch.createStarted();
      double[] thetasCS = classifierApprox.learn(dataSet, Regularization.l2(0.001));
      watchCS.stop();
      System.out.println("PREDICTING using exact F1 score ... ");
      List<Prediction<LinearChain>> predictionsCS = classifierExact.predict(testDataSet,thetasCS);
      Triple<Double, Double, Double> precisionRecallF1CS = evaluate(predictionsCS, targetTag, testDataSet.getIndicesByTag());

      System.out.println("Trained Classifier:\t CostSensitiveApproximatedF1 (~)");
      System.out.println("targetTag:\t" + targetTag);
      System.out.println("Precision:\t" + precisionRecallF1CS.getLeft() );
      System.out.println("Recall:\t" + precisionRecallF1CS.getMiddle());
      System.out.println("F1:\t" + precisionRecallF1CS.getRight());
      System.out.println("time (~):\t" + watchCS.elapsed(TimeUnit.SECONDS));
      System.out.println();
    }
  }
  @Ignore
  @Test
  public void testLearnAndPredictAndCompare() throws Exception {

    System.out.println("TESTING: " + testName.getMethodName());
    File file = new File(TestLinearChainApproxF1Classifier.class
          .getResource("TestLinearChainF1.300.train").toURI());
    File testfile = new File(TestLinearChainApproxF1Classifier.class
          .getResource("TestLinearChainF1.300.test").toURI());

    LinearChainDataSet dataSet = LinearChainDataSet.loadFromFile(file);
    LinearChainDataSet testDataSet = LinearChainDataSet.loadFromFile(testfile);

    int[] targetTags = new int[] {1};

    for (int targetTag : targetTags) {
      MPGLinearChainClassifer classifierCS =
          new MPGLinearChainClassifer(LinearChainApproxF1.class, targetTag);
      MPGLinearChainClassifer classifier2 =
          new MPGLinearChainClassifer(LinearChainF1.class, targetTag);


      System.out.println("TRAINING a model with approximated F1 score");
      Stopwatch watchCS = Stopwatch.createStarted();
      double[] thetasCS = classifierCS.learn(dataSet, Regularization.l2(0.001));
      watchCS.stop();
      System.out.println("PREDICTING using exact F1 score ... ");
      List<Prediction<LinearChain>> predictionsCS = classifier2.predict(testDataSet,thetasCS);
      Triple<Double, Double, Double> precisionRecallF1CS = evaluate(predictionsCS, targetTag, testDataSet.getIndicesByTag());

      System.out.println("Trained Classifier:\t CostSensitiveApproximatedF1 (~)");
      System.out.println("targetTag:\t" + targetTag);
      System.out.println("Precision:\t" + precisionRecallF1CS.getLeft());
      System.out.println("Recall:\t" + precisionRecallF1CS.getMiddle());
      System.out.println("F1:\t" + precisionRecallF1CS.getRight());
      System.out.println("time (~) (*):\t" + watchCS.elapsed(TimeUnit.SECONDS));
      System.out.println();
    }
  }


  protected static Triple<Double, Double, Double> evaluate(
          List<Prediction<LinearChain>> predictions, Integer targetTag,
          Map<Integer, Integer> indicesByEffectiveTag) {
    if (predictions == null) {
      return Triple.of(Double.NaN, Double.NaN, Double.NaN);
    }

    int numOfTags = indicesByEffectiveTag.size();
    int[][] confusionMatrix = new int[numOfTags][numOfTags];

    for (Prediction<LinearChain> prediction : predictions) {
      int[] goldenTags = prediction.getGoldenPermutation().getTagSequence();
      int[] predictedTags = prediction.getPredictionPermutation().getTagSequence();
      Assert.isTrue(goldenTags.length == predictedTags.length);

      for (int index = 0; index < goldenTags.length; index++) {
        Integer goldenIndex = indicesByEffectiveTag.get(goldenTags[index]);
        Integer predictedIndex = indicesByEffectiveTag.get(predictedTags[index]);
        confusionMatrix[goldenIndex][predictedIndex] += 1;
      }
    }

    Integer index = indicesByEffectiveTag.get(targetTag);
    double precision = precision(confusionMatrix, index);
    double recall = recall(confusionMatrix, index);
    double f1 = f1(precision, recall);
    return Triple.of(precision, recall, f1);
  }

  private static double f1(double precision, double recall) {
    if ((precision + recall) == 0) {
      return 0;
    }
    return 2 * precision * recall / (precision + recall);
  }

  protected static double precision(int[][] confusionMatrix, int classIndex) {
    double correct = 0;
    double total = 0;
    for (int i = 0; i < confusionMatrix.length; i++) {
      if (i == classIndex) {
        correct += confusionMatrix[i][classIndex];
      }
      total += confusionMatrix[i][classIndex];
    }
    if (total == 0) {
      return 0;
    }
    return correct / total;
  }

  protected static double recall(int[][] confusionMatrix, int classIndex) {
    double correct = 0;
    double total = 0;
    for (int j = 0; j < confusionMatrix.length; j++) {
      if (j == classIndex) {
        correct += confusionMatrix[classIndex][j];
      }
      total += confusionMatrix[classIndex][j];
    }
    if (total == 0) {
      return 0;
    }
    return correct / total;
  }
}
