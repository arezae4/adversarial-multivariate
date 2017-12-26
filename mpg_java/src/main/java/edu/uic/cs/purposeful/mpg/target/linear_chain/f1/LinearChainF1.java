package edu.uic.cs.purposeful.mpg.target.linear_chain.f1;

import java.util.Arrays;
import java.util.BitSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.primitives.Doubles;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.common.Misc;
import edu.uic.cs.purposeful.mpg.learning.linear_chain.LinearChainDataSet.LinearChainDataSetInstance;
import edu.uic.cs.purposeful.mpg.target.OptimizationTarget;
import edu.uic.cs.purposeful.mpg.target.common.GeneralFMeasureMaximizer;
import edu.uic.cs.purposeful.mpg.target.linear_chain.LinearChain;
import edu.uic.cs.purposeful.mpg.target.linear_chain.LinearChainConfig;
import net.mintern.primitive.pair.IntPair;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;
import no.uib.cipr.matrix.sparse.SparseVector;

public class LinearChainF1
    implements OptimizationTarget<LinearChain, Pair<Integer, LinearChainDataSetInstance>> {
  private Integer targetTag;
  private Integer oneNonTargetTag;

  private Map<Integer, Integer> indicesByTag;
  private Map<IntPair, Integer> indicesByTagPair;
  private int numOfTags;
  private int numOfTagPairs;

  private CompRowMatrix unigramFeatureMatrix;
  private CompRowMatrix bigramFeatureMatrix;

  private CompRowMatrix unigramTransposedFeatureMatrix;
  private CompRowMatrix bigramTransposedFeatureMatrix;

  private int numOfUnigramFeatures;
  private int numOfBigramFeatures;

  private int numOfPositions;

  private int totalNumOfUnigramWeights;
  private int totalNumOfBigramWeights;
  private int totalNumOfWeights;

  private LinearChain goldenLinearChain;
  private Vector goldenFeatureValues;

  private GeneralFMeasureMaximizer gfm;

  @Override
  public void initialize(Pair<Integer, LinearChainDataSetInstance> initialData,
      boolean duringTraining) {
    this.targetTag = initialData.getLeft();

    LinearChainDataSetInstance instance = initialData.getRight();

    this.indicesByTag = instance.getIndicesByTag();
    Assert.isTrue(indicesByTag.containsKey(targetTag), String
        .format("targetTag=[%d] is not among data set tags=%s", targetTag, indicesByTag.keySet()));
    this.indicesByTagPair = instance.getIndicesByTagPair();

    this.numOfTags = indicesByTag.size();
    this.numOfTagPairs = indicesByTagPair.size();

    // we use the first non-targetTag as non-targetTag
    this.oneNonTargetTag = getFirstNonTargetTag();

    this.unigramFeatureMatrix = instance.getUnigramFeatureMatrix();
    this.bigramFeatureMatrix = instance.getBigramFeatureMatrix();

    this.unigramTransposedFeatureMatrix = instance.getUnigramTransposedFeatureMatrix();
    this.bigramTransposedFeatureMatrix = instance.getBigramTransposedFeatureMatrix();

    this.numOfUnigramFeatures = unigramFeatureMatrix.numColumns();
    this.numOfBigramFeatures = bigramFeatureMatrix.numColumns();

    this.totalNumOfUnigramWeights = numOfUnigramFeatures * numOfTags;
    this.totalNumOfBigramWeights = numOfBigramFeatures * numOfTagPairs;
    this.totalNumOfWeights = totalNumOfUnigramWeights + totalNumOfBigramWeights;

    int[] goldenTags = instance.getGoldenTags();
    this.numOfPositions = goldenTags.length;
    Assert.isTrue(this.numOfPositions == unigramFeatureMatrix.numRows());
    Assert.isTrue(this.numOfPositions == bigramFeatureMatrix.numRows());

    goldenLinearChain = new LinearChain(goldenTags, indicesByTag, indicesByTagPair);
    this.gfm =
        new GeneralFMeasureMaximizer(targetTag, indicesByTag, indicesByTagPair, numOfPositions);

    if (duringTraining) {
      goldenFeatureValues = buildGoldenFeatureValues();
    }
  }

  private Integer getFirstNonTargetTag() {
    Integer nonTargetTag = null;
    for (Integer effectiveTag : indicesByTag.keySet()) {
      if (effectiveTag != targetTag) {
        nonTargetTag = effectiveTag;
        break;
      }
    }
    Assert.notNull(nonTargetTag);
    return nonTargetTag;
  }

  private Vector buildGoldenFeatureValues() {
    // #positions * #tags
    LinkedSparseMatrix unigramExistenceMatrix = goldenLinearChain.getUnigramExistenceMatrix();
    // #unigramFeatures * #tags
    LinkedSparseMatrix unigramFeatureValueMatrix =
        new LinkedSparseMatrix(numOfUnigramFeatures, numOfTags);
    // (#positions * #unigramFeatures)^T * (#positions * #tags)
    // = (#unigramFeatures * #positions) * (#positions * #tags) = (#unigramFeatures * #tags)
    // unigramFeatureMatrix.transAmult(unigramExistenceMatrix, unigramFeatureValueMatrix);
    unigramTransposedFeatureMatrix.mult(unigramExistenceMatrix, unigramFeatureValueMatrix);

    // #positions * #pairs
    LinkedSparseMatrix bigramExistenceMatrix = goldenLinearChain.getBigramExistenceMatrix();
    // #bigramFeatures * #pairs
    LinkedSparseMatrix bigramFeatureValueMatrix =
        new LinkedSparseMatrix(numOfBigramFeatures, numOfTagPairs);
    // (#positions * #bigramFeatures)^T * (#positions * #pairs)
    // = (#bigramFeatures * #positions) * (#positions * #pairs) = (#bigramFeatures * #pairs)
    // bigramFeatureMatrix.transAmult(bigramExistenceMatrix, bigramFeatureValueMatrix);
    bigramTransposedFeatureMatrix.mult(bigramExistenceMatrix, bigramFeatureValueMatrix);

    return toVector(unigramFeatureValueMatrix, bigramFeatureValueMatrix);
  }

  private Vector toVector(LinkedSparseMatrix unigramFeatureValueMatrix,
      LinkedSparseMatrix bigramFeatureValueMatrix) {
    Vector goldenFeatureValues = new SparseVector(totalNumOfWeights);
    for (MatrixEntry entry : unigramFeatureValueMatrix) {
      int index = entry.column() * numOfUnigramFeatures + entry.row();
      goldenFeatureValues.set(index, entry.get());
    }
    for (MatrixEntry entry : bigramFeatureValueMatrix) {
      int index = totalNumOfUnigramWeights + entry.column() * numOfBigramFeatures + entry.row();
      goldenFeatureValues.set(index, entry.get());
    }
    return goldenFeatureValues;
  }

  @Override
  public Vector computeExpectedFeatureValues(double[] probabilities,
      LinkedHashSet<LinearChain> permutations) {
    // #positions * #tags
    LinkedSparseMatrix unigramMarginalProbabilityMatrix =
        new LinkedSparseMatrix(numOfPositions, numOfTags);
    // #positions * #pairs
    LinkedSparseMatrix bigramMarginalProbabilityMatrix =
        new LinkedSparseMatrix(numOfPositions, numOfTagPairs);

    int permutationIndex = 0;
    for (LinearChain permutation : permutations) {
      double probability = probabilities[permutationIndex++];
      if (Misc.roughlyEquals(probability, 0)) {
        continue; // permutation has no contribution, skip
      }
      unigramMarginalProbabilityMatrix.add(probability, permutation.getUnigramExistenceMatrix());
      bigramMarginalProbabilityMatrix.add(probability, permutation.getBigramExistenceMatrix());
    }

    // #unigramFeatures * #tags
    LinkedSparseMatrix unigramExpectedFeatureValueMatrix =
        new LinkedSparseMatrix(numOfUnigramFeatures, numOfTags);
    // (#positions * #unigramFeatures)^T * (#positions * #tags)
    // = (#unigramFeatures * #positions) * (#positions * #tags) = (#unigramFeatures * #tags)
    // unigramFeatureMatrix.transAmult(unigramMarginalProbabilityMatrix,
    // unigramExpectedFeatureValueMatrix);
    unigramTransposedFeatureMatrix.mult(unigramMarginalProbabilityMatrix,
        unigramExpectedFeatureValueMatrix);

    // #bigramFeatures * #pairs
    LinkedSparseMatrix bigramExpectedFeatureValueMatrix =
        new LinkedSparseMatrix(numOfBigramFeatures, numOfTagPairs);
    // (#positions * #bigramFeatures)^T * (#positions * #pairs)
    // = (#bigramFeatures * #positions) * (#positions * #pairs) = (#bigramFeatures * #pairs)
    // bigramFeatureMatrix.transAmult(bigramMarginalProbabilityMatrix,
    // bigramExpectedFeatureValueMatrix);
    bigramTransposedFeatureMatrix.mult(bigramMarginalProbabilityMatrix,
        bigramExpectedFeatureValueMatrix);

    return toVector(unigramExpectedFeatureValueMatrix, bigramExpectedFeatureValueMatrix);
  }

  @Override
  public double computeScore(LinearChain maximizerPermutation, LinearChain minimizerPermutation) {
    Assert.isTrue(maximizerPermutation.getLength() == minimizerPermutation.getLength());
    Assert.isTrue(maximizerPermutation.getLength() == numOfPositions);
    int totalNumOfTargetTags =
        maximizerPermutation.countTag(targetTag) + minimizerPermutation.countTag(targetTag);
    if (totalNumOfTargetTags == 0) {
      return 1.0;
    }

    int intersectedNumOfTargetTags =
        maximizerPermutation.countCommonTag(targetTag, minimizerPermutation);

    return 2.0 * intersectedNumOfTargetTags / totalNumOfTargetTags;
  }

  @Override
  public Set<LinearChain> getInitialMaximizerPermutations() {
    LinkedHashSet<LinearChain> resutls = new LinkedHashSet<>();
    for (int effectiveTag : indicesByTag.keySet()) {
      int[] sequence = new int[numOfPositions];
      Arrays.fill(sequence, effectiveTag);
      resutls.add(new LinearChain(sequence, indicesByTag, indicesByTagPair));
    }
    return resutls;
  }

  @Override
  public Set<LinearChain> getInitialMinimizerPermutations() {
    // same as #getInitialMaximizerPermutations()
    return getInitialMaximizerPermutations();
  }

  @Override
  public double[] computeLagrangePotentials(double[] thetasInOneArray) {
    // first part is for unigram
    double[] unigramThetas = ArrayUtils.subarray(thetasInOneArray, 0, totalNumOfUnigramWeights);
    // #unigramFeatures * #tags
    DenseMatrix unigramThetaMatrix =
        new DenseMatrix(numOfUnigramFeatures, numOfTags, unigramThetas, false);
    // #positions * #tags
    DenseMatrix unigramLagrangePotentialMatrix = new DenseMatrix(numOfPositions, numOfTags);
    // (#positions * #unigramFeatures) * (#unigramFeatures * #tags) = (#positions * #tags)
    unigramFeatureMatrix.mult(unigramThetaMatrix, unigramLagrangePotentialMatrix);
    double[] unigramLagrangePotentials = unigramLagrangePotentialMatrix.getData();

    // second part is for bigram
    double[] bigramThetas =
        ArrayUtils.subarray(thetasInOneArray, totalNumOfUnigramWeights, thetasInOneArray.length);
    // #bigramFeatures * #pairs
    DenseMatrix bigramThetaMatrix =
        new DenseMatrix(numOfBigramFeatures, numOfTagPairs, bigramThetas, false);
    // #positions * #pairs
    DenseMatrix bigramLagrangePotentialMatrix = new DenseMatrix(numOfPositions, numOfTagPairs);
    // (#positions * #bigramFeatures) * (#bigramFeatures * #pairs) = (#positions * #pairs)
    bigramFeatureMatrix.mult(bigramThetaMatrix, bigramLagrangePotentialMatrix);
    double[] bigramLagrangePotentials = bigramLagrangePotentialMatrix.getData();

    return Doubles.concat(unigramLagrangePotentials, bigramLagrangePotentials);
  }

  /**
   * @see GeneralFMeasureMaximizer#retrieveLagrangePotential(int, int, int, double[])
   */
  @Override
  public double aggregateLagrangePotentials(LinearChain minimizerPermutation,
      double[] lagrangePotentials) {
    DenseMatrix lagrangePotentialMatrix =
        new DenseMatrix(numOfPositions, numOfTags + numOfTagPairs, lagrangePotentials, false);

    int[] unigramIndices = minimizerPermutation.getUnigramIndices();
    int[] bigramIndices = minimizerPermutation.getBigramIndices();

    double sum = 0.0;
    for (int positionIndex = 0; positionIndex < numOfPositions; positionIndex++) {
      double unigramPotential =
          lagrangePotentialMatrix.get(positionIndex, unigramIndices[positionIndex]);

      double bigramPotential = 0;
      if (positionIndex != 0 || LinearChainConfig.ADD_STARTING_TAG) {
        int bigramIndex = bigramIndices[positionIndex];
        Assert.isTrue(bigramIndex >= 0);
        bigramPotential = lagrangePotentialMatrix.get(positionIndex, numOfTags + bigramIndex);
      }

      sum += (unigramPotential + bigramPotential);
    }
    return sum;
  }

  @VisibleForTesting
  Pair<Double, LinkedSparseMatrix> computeMarginalProbabilityMatrixP(double[] probabilities,
      LinkedHashSet<LinearChain> permutations) {
    double p0 = 0.0;
    LinkedSparseMatrix matrixP = new LinkedSparseMatrix(numOfPositions, numOfPositions);

    int permutationIndex = 0;
    for (LinearChain permutation : permutations) {
      double probability = probabilities[permutationIndex++];
      if (Misc.roughlyEquals(probability, 0)) {
        continue;
      }

      BitSet existenceSequence = permutation.getExistenceSequence(targetTag);
      int numOfOnes = existenceSequence.cardinality(); // s
      if (numOfOnes == 0) {
        p0 += probability;
      } else {
        for (int index = existenceSequence.nextSetBit(0); index >= 0; index =
            existenceSequence.nextSetBit(index + 1)) { // i
          // the probability that there are total s '1's, and i-th position is '1'
          matrixP.add(index, numOfOnes - 1, probability);
        }
      }
    }

    return Pair.of(p0, matrixP);
  }

  @Override
  public Pair<LinearChain, Double> findBestMaximizerResponsePermutation(
      double[] minimizerProbabilities, LinkedHashSet<LinearChain> existingMinimizerPermutations,
      double[] lagrangePotentials) {
    Pair<Double, LinkedSparseMatrix> p0AndMatrixP =
        computeMarginalProbabilityMatrixP(minimizerProbabilities, existingMinimizerPermutations);
    double p0 = p0AndMatrixP.getLeft();
    LinkedSparseMatrix matrixP = p0AndMatrixP.getRight();
    Pair<BitSet, Double> bestMaximizerResponse = gfm.gfm(p0, matrixP);

    double lagrangePotentialsExpectation = computeLagrangePotentialsExpectation(
        minimizerProbabilities, existingMinimizerPermutations, lagrangePotentials);
    double bestResponseValue = bestMaximizerResponse.getRight() - lagrangePotentialsExpectation;

    LinearChain bestResponsePermutation = buildLinearChain(bestMaximizerResponse.getLeft());

    return Pair.of(bestResponsePermutation, bestResponseValue);
  }

  private LinearChain buildLinearChain(BitSet binarySequence) {
    int[] bestResponseTags = new int[numOfPositions];
    for (int index = binarySequence.nextSetBit(0); index >= 0; index =
        binarySequence.nextSetBit(index + 1)) {
      bestResponseTags[index] = targetTag;
    }
    for (int index = 0; index < bestResponseTags.length; index++) {
      if (bestResponseTags[index] != targetTag) {
        bestResponseTags[index] = oneNonTargetTag;
      }
    }
    return new LinearChain(bestResponseTags, indicesByTag, indicesByTagPair);
  }

  private double computeLagrangePotentialsExpectation(double[] minimizerProbabilities,
      LinkedHashSet<LinearChain> existingMinimizerPermutations, double[] lagrangePotentials) {
    double lagrangePotentialsExpectation = 0;
    int permutationIndex = 0;
    for (LinearChain permutation : existingMinimizerPermutations) {
      double probability = minimizerProbabilities[permutationIndex++];
      if (Misc.roughlyEquals(probability, 0)) {
        continue;
      }
      double lagrangePotential = aggregateLagrangePotentials(permutation, lagrangePotentials);
      lagrangePotentialsExpectation += probability * lagrangePotential;
    }
    return lagrangePotentialsExpectation;
  }

  @Override
  public Pair<LinearChain, Double> findBestMinimizerResponsePermutation(
      double[] maximizerProbabilities, LinkedHashSet<LinearChain> existingMaximizerPermutations,
      double[] lagrangePotentials) {
    Pair<Double, LinkedSparseMatrix> p0AndMatrixP =
        computeMarginalProbabilityMatrixP(maximizerProbabilities, existingMaximizerPermutations);
    double p0 = p0AndMatrixP.getLeft();
    LinkedSparseMatrix matrixP = p0AndMatrixP.getRight();
    return gfm.linearChainGFM(p0, matrixP, lagrangePotentials);
  }

  @Override
  public boolean isLegalMaximizerPermutation(LinearChain permutation) {
    return permutation != null;
  }

  @Override
  public boolean isLegalMinimizerPermutation(LinearChain permutation) {
    return permutation != null;
  }

  @Override
  public LinearChain getGoldenPermutation() {
    return goldenLinearChain;
  }

  @Override
  public Vector getGoldenFeatureValues() {
    return goldenFeatureValues;
  }

}
