package edu.uic.cs.purposeful.mpg.target.linear_chain.f1;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import edu.uic.cs.purposeful.mpg.MPGConfig;
import gnu.trove.list.array.TIntArrayList;
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

public class LinearChainApproxF1
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
  
  private final double discretization = .1;

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

   private double computeScore(LinearChain maximizerPermutation, LinearChain minimizerPermutation, int toIndex) {
    Assert.isTrue(maximizerPermutation.getLength() == minimizerPermutation.getLength());
    Assert.isTrue(maximizerPermutation.getLength() == numOfPositions);
    int totalNumOfTargetTags =
        maximizerPermutation.countTag(targetTag,toIndex) + minimizerPermutation.countTag(targetTag,toIndex);
    if (totalNumOfTargetTags == 0) {
      return 1.0;
    }

    int intersectedNumOfTargetTags =
        maximizerPermutation.countCommonTag(targetTag, minimizerPermutation,toIndex);

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

  private double aggregateLagrangePotentials(LinearChain minimizerPermutation,
      double[] lagrangePotentials, int toIndex) {
    DenseMatrix lagrangePotentialMatrix =
        new DenseMatrix(toIndex, numOfTags + numOfTagPairs, lagrangePotentials, false);

    int[] unigramIndices = minimizerPermutation.getUnigramIndices();
    int[] bigramIndices = minimizerPermutation.getBigramIndices();

    double sum = 0.0;
    for (int positionIndex = 0; positionIndex < toIndex; positionIndex++) {
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

  //O(Nn)
  @VisibleForTesting
  DenseMatrix computeMarginalProbabilityOfTagsPerPosition(double[] probabilities,
      LinkedHashSet<LinearChain> permutations) {

    DenseMatrix matrixP = new DenseMatrix(numOfPositions, 2); // 0: non-targetTag , 1: targetTag
    //LinkedSparseMatrix matrixP = new LinkedSparseMatrix(numOfPositions, indicesByTag.size());

    int permutationIndex = 0;
    for (LinearChain permutation : permutations) {
      double probability = probabilities[permutationIndex++];
      if (Misc.roughlyEquals(probability, 0)) {
        continue;
      }
      for(int pos = 0; pos < numOfPositions; pos++){
    	  int tag = permutation.getTagSequence()[pos];
    	  //int tagIdx = indicesByTag.get(tag);
	      //matrixP.add(pos, tagIdx, probability);
        matrixP.add(pos, (tag == targetTag ? 1 : 0), probability);
      }
    }
    return matrixP;
  }

  DenseMatrix computeMarginalCostForTagsPerPosition(DenseMatrix matrixP, double t){
  	DenseMatrix costMatrix= new DenseMatrix(numOfPositions,2);
	/*for(int i = 0; i < numOfPositions; i++){
		double targetTagProbability = matrixP.get(i,indicesByTag.get(targetTag));
		if (!Misc.roughlyEquals(1-targetTagProbability, 0)) {
			costMatrix.set(i,0, (1-targetTagProbability) * (t)); // False Positive: choosing target tag
		}
		if (!Misc.roughlyEquals(targetTagProbability, 0)) {
			costMatrix.set(i,1, (targetTagProbability) * (2-t)); // False Negative: not choosing target tag
		}
	}*/
	  DenseMatrix T = new DenseMatrix(new double[][]{{t,0},{0,(2-t)}});
    matrixP.multAdd(1,T,costMatrix);
	  return costMatrix;
  }

  // O(Tnm^2  + Nn)
  @Override
  public Pair<LinearChain, Double> findBestMaximizerResponsePermutation(
	      double[] minimizerProbabilities, LinkedHashSet<LinearChain> existingMinimizerPermutations,
	      double[] lagrangePotentials) {
/*
	    Pair<LinearChain, Double> bestMaximizerResponse= costSensitiveCostMinimizer(existingMinimizerPermutations,minimizerProbabilities);
	   LinearChain bestResponsePermutation = bestMaximizerResponse.getLeft();
	    double bestResponseF1Value = 0;
	    int permutationIndex = 0;
	    for(LinearChain permutation : existingMinimizerPermutations){//O(Nn)
		if (Misc.roughlyEquals(minimizerProbabilities[permutationIndex], 0)) {
				    permutationIndex++;
			            continue;
		}
	    	bestResponseF1Value += minimizerProbabilities[permutationIndex++] * 
			(computeScore(permutation, bestResponsePermutation)- aggregateLagrangePotentials(permutation, lagrangePotentials));
	    } 
	    double bestResponseValue = bestResponseF1Value;
	    return Pair.of(bestResponsePermutation, bestResponseValue);
*/		double lagrangePotentialExpectation = computeLagrangePotentialsExpectation(minimizerProbabilities,existingMinimizerPermutations,lagrangePotentials);
		return costSensitiveCostMinimizer(existingMinimizerPermutations,minimizerProbabilities, lagrangePotentialExpectation); 
	  }

  /**
   * Computes the approximated best response for maximizer given the current minimizer distribution in O(T4n + TnN)
   */
  private Pair<LinearChain, Double> costSensitiveCostMinimizer(LinkedHashSet<LinearChain> existingMinimizerPermutations, double[] minimizerProbabilities, double lagrangePotentialExpectation){
	 double memo[] = new double[2];
	 Arrays.fill(memo,0);
	 DenseMatrix matrixP = computeMarginalProbabilityOfTagsPerPosition(minimizerProbabilities, existingMinimizerPermutations); //O(Nn)
	 LinearChain bestTags = null;
	 double bestExpectedScore = Double.NEGATIVE_INFINITY;
	 double bestT = 0;
	 for(double t = 0.1; t <=1; t += this.discretization){
		BitSet bestTagsForT = new BitSet(numOfPositions);
		DenseMatrix marginalCostSensitiveCostMatrix =computeMarginalCostForTagsPerPosition(matrixP, t); // O(n)
		for( int pos = 0; pos < numOfPositions ; pos++){
			double tagCostAtPosForTargetTag = marginalCostSensitiveCostMatrix.get(pos, 0);
			double tagCostAtPosForNonTargetTag = marginalCostSensitiveCostMatrix.get(pos, 1);
			boolean targetOrNot = tagCostAtPosForTargetTag < tagCostAtPosForNonTargetTag;
			bestTagsForT.set(pos, targetOrNot);
        }
		//System.out.format("t: %.2f, cost: %.3f\n",t,bestCostForT);
        double expectedScoreForT = 0;
        int permutationIndex = 0;
        LinearChain bestTagsLinearChain = buildLinearChain(bestTagsForT);
        for(LinearChain permutation : existingMinimizerPermutations){//O(Nn)
            if (Misc.roughlyEquals(minimizerProbabilities[permutationIndex], 0)) {
                permutationIndex++;
                continue;
            }
            expectedScoreForT += minimizerProbabilities[permutationIndex++] *
            computeScore(permutation, bestTagsLinearChain);
        }
        expectedScoreForT -= lagrangePotentialExpectation;
        if(expectedScoreForT > bestExpectedScore){
            bestExpectedScore = expectedScoreForT;
            bestTags = buildLinearChain(bestTagsForT);
            bestT = t;
        }
	 }
	 Assert.isFalse(bestTags == null);
   if(MPGConfig.SHOW_RUNNING_TRACING) {
     if (bestT != 0) System.out.format("Best t for maximizer = %.3f\n", bestT);
   }
	 return Pair.of(bestTags, bestExpectedScore);
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

  // O(Nn)
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

  // O(N)
  private double computeCostSensitiveCost(int currentPositionIndex, int predictedTag, LinkedHashSet<LinearChain> existingPermutations,
		  double[] permutationProbabilites, double weight_t) {
  	double cost = 0.0;
	int permutationIndex = 0;
	for(LinearChain permutation : existingPermutations){
		if(permutation.getTagSequence()[currentPositionIndex] == targetTag && predictedTag != targetTag)
			cost += permutationProbabilites[permutationIndex] * (2 - weight_t); // false negative;
		else if(permutation.getTagSequence()[currentPositionIndex] != targetTag && predictedTag == targetTag)
			cost += permutationProbabilites[permutationIndex] * weight_t; // false positive
		permutationIndex++;

	}
	return cost;
  }
  /**
   * Computes the approximated best response for minimizer given the current maximizer distribution in O(Tnm^2 + TnN)
   */
  private Pair<LinearChain, Double> costSensitiveCostMaximizer(LinkedHashSet<LinearChain> existingMaximizerPermutations,
		  double[] maximizerProbabilities, double[] lagrangePotentials){
	 DenseMatrix matrixP = computeMarginalProbabilityOfTagsPerPosition(maximizerProbabilities, existingMaximizerPermutations); // O(Nn)
	 double bestT = 0.0;
	 double bestExpectedScore = Double.MAX_VALUE;
	 int bestTags[] = new int[numOfPositions];
	 int startTag = LinearChainConfig.SEQUENCE_STARTING_TAG;

     for(double t = 0.1 ; t <= 1; t += this.discretization){

        double cache[][] = new double[numOfPositions][indicesByTag.size()];
        int previousTagCache[][] = new int[numOfPositions][indicesByTag.size()];
        TIntArrayList bestTagsForT = new TIntArrayList(numOfPositions);
		DenseMatrix marginalCostSensitiveCostMatrix =computeMarginalCostForTagsPerPosition(matrixP, t); // O(n)
        double bestSumForCurrentT = Double.NEGATIVE_INFINITY;
        int bestLastPositionTagForT = -Integer.MAX_VALUE;
        for( int pos = 0; pos < numOfPositions ; pos++) {
            for (int currentTag : indicesByTag.keySet()) {
                int currentTagIdx = indicesByTag.get(currentTag);
                double bestSumPrevious = Double.NEGATIVE_INFINITY;
                double tagCostAtPos = (currentTag == targetTag) ? marginalCostSensitiveCostMatrix.get(pos, 0) : marginalCostSensitiveCostMatrix.get(pos, 1);
                for (int previousTag : indicesByTag.keySet()) {
                    int previousTagIdx = indicesByTag.get(previousTag);
                    double potential = retrieveLagrangePotentialNormalized(pos == 0 ? startTag : previousTag, currentTag, pos, lagrangePotentials);
                    double tmp = (pos == 0 ? 0.0 : cache[pos - 1][previousTagIdx]) + potential;
                    if (tmp > bestSumPrevious) {
                        bestSumPrevious = tmp;
                        if (pos > 0) {
                            previousTagCache[pos][currentTagIdx] = previousTag;
                        }
                    }
                    if (pos == 0)
                        break; // no further redundant iteration;
                }
                cache[pos][currentTagIdx] = bestSumPrevious + tagCostAtPos/numOfPositions;
                if ( bestSumForCurrentT == Double.NEGATIVE_INFINITY || cache[pos][currentTagIdx] > bestSumForCurrentT) {
                    bestSumForCurrentT = cache[pos][currentTagIdx];
                    bestLastPositionTagForT = currentTag;
                }
            }
        }
        Assert.isTrue(bestLastPositionTagForT != -Integer.MAX_VALUE);
        bestTagsForT.add(bestLastPositionTagForT);
        int lastTagIdx = indicesByTag.get(bestLastPositionTagForT);
        for (int i = numOfPositions - 1; i >= 1; i--) {
            int prevTag = previousTagCache[i][lastTagIdx];
            bestTagsForT.insert(0, prevTag);
            lastTagIdx = indicesByTag.get(prevTag);
        }
        double expectedScoreForT = computeExpectedScore(new LinearChain(bestTagsForT.toArray(),indicesByTag,indicesByTagPair),
                maximizerProbabilities,existingMaximizerPermutations,lagrangePotentials, numOfPositions); //O(Nn)
        if(expectedScoreForT < bestExpectedScore){
            bestExpectedScore = expectedScoreForT;
            System.arraycopy(bestTagsForT.toArray(), 0, bestTags, 0, numOfPositions);
            bestT = t;
            //System.out.format("Update t: %.2f, score: %.3f\n",bestT ,bestExpectedScore);
        }

     }
     if(MPGConfig.SHOW_RUNNING_TRACING) {
       if (bestT != 0) System.out.format("Best t for minimizer:%.3f\n", bestT);
     }
     return Pair.of(new LinearChain(bestTags, indicesByTag, indicesByTagPair), bestExpectedScore);
 }	 


 /**
  * @see LinearChainF1#aggregateLagrangePotentials(LinearChain, double[])
  */
  private double retrieveLagrangePotential(int previousTag, int currentTag,
      int currentPositionIndex, double[] lagrangePotentials) {
    Assert.isTrue(
    	currentPositionIndex != 0 || previousTag == LinearChainConfig.SEQUENCE_STARTING_TAG);

    int numOfColumns = indicesByTag.size() + indicesByTagPair.size();
    DenseMatrix lagrangePotentialMatrix =
       new DenseMatrix(numOfPositions, numOfColumns, lagrangePotentials, false);

    Integer tagIndex = indicesByTag.get(currentTag);
    double unigramPotential = lagrangePotentialMatrix.get(currentPositionIndex, tagIndex);

    double bigramPotential = 0;
    if (currentPositionIndex != 0 || LinearChainConfig.ADD_STARTING_TAG) {
    	Integer tagPairIndex = indicesByTagPair.get(IntPair.of(previousTag, currentTag));
    	Assert.notNull(tagPairIndex);
    	bigramPotential =
    			lagrangePotentialMatrix.get(currentPositionIndex, indicesByTag.size() + tagPairIndex);
    }

   return unigramPotential + bigramPotential;
  }

  /**
  * @see LinearChainF1#aggregateLagrangePotentials(LinearChain, double[])
  */
  private double retrieveLagrangePotentialNormalized(int previousTag, int currentTag,
      int currentPositionIndex, double[] lagrangePotentials) {
    Assert.isTrue(
    	currentPositionIndex != 0 || previousTag == LinearChainConfig.SEQUENCE_STARTING_TAG);

    int numOfColumns = indicesByTag.size() + indicesByTagPair.size();
    DenseMatrix lagrangePotentialMatrix =
       new DenseMatrix(numOfPositions, numOfColumns, lagrangePotentials, false);

    Integer tagIndex = indicesByTag.get(currentTag);
    double unigramPotential = lagrangePotentialMatrix.get(currentPositionIndex, tagIndex);

    double bigramPotential = 0;
    if (currentPositionIndex != 0 || LinearChainConfig.ADD_STARTING_TAG) {
    	Integer tagPairIndex = indicesByTagPair.get(IntPair.of(previousTag, currentTag));
    	Assert.notNull(tagPairIndex);
    	bigramPotential =
    			lagrangePotentialMatrix.get(currentPositionIndex, indicesByTag.size() + tagPairIndex);
    }

    double maxUnigram = Double.NEGATIVE_INFINITY;
    double maxBigram = Double.NEGATIVE_INFINITY;
    double minUnigram = Double.MAX_VALUE;
    double minBigram = Double.MAX_VALUE;
    assert(lagrangePotentials.length == numOfPositions * (numOfTags + numOfTagPairs));
    for(int i = 0; i < lagrangePotentials.length; i++){
        if( i < numOfPositions * numOfTags && lagrangePotentials[i] > maxUnigram)
            maxUnigram = lagrangePotentials[i];
        if( i < numOfPositions * numOfTags && lagrangePotentials[i] < minUnigram)
          minUnigram = lagrangePotentials[i];
        if( i >= numOfPositions * numOfTags && lagrangePotentials[i] > maxBigram)
            maxBigram = lagrangePotentials[i];
        if( i >= numOfPositions * numOfTags && lagrangePotentials[i] < minBigram)
            minBigram = lagrangePotentials[i];
    }
    //return (unigramPotential + bigramPotential) / Collections.max(Arrays.asList(ArrayUtils.toObject(lagrangePotentials)));
	//System.out.format(" potentials : %.4f, %.4f, %.4f, %.4f\n", (unigramPotential - minUnigram)/(maxUnigram - minUnigram), minBigram, maxUnigram, maxBigram);
    return ((unigramPotential - minUnigram) / (maxUnigram - minBigram)) + ((bigramPotential - minBigram)/ (maxBigram - minBigram));
  }

  /**
   * O(N+n)
   */
  private double computeLagrangePotentialExpectationAtPosition(double[] maximizerProbabilities, 
		  int previousTag, int currentTag, int currentPositionIndex, double[] lagrangePotentials) {
  	double lagrangePotentialAtPosition = retrieveLagrangePotential(previousTag,currentTag,currentPositionIndex,lagrangePotentials);
	double result = 0.0;
	for(int i = 0; i < maximizerProbabilities.length ; i++){
		result += maximizerProbabilities[i] * lagrangePotentialAtPosition;
	}
	return result;
  }

  // O(Nn)
  private double computeExpectedScore( LinearChain minimizerPermutation, double[] maximizerProbabilities,
                                LinkedHashSet<LinearChain> existingMaximizerPermutations,
                                double[] lagrangePotentials, int toIndex){
      int permutationIndex = 0;
      double result = 0;
      for(LinearChain permutation : existingMaximizerPermutations){ //O(N)
          if (Misc.roughlyEquals(maximizerProbabilities[permutationIndex], 0)) {
				    permutationIndex++;
			            continue;
		}
	    result += maximizerProbabilities[permutationIndex++] * computeScore(permutation, minimizerPermutation, toIndex);
    }

    result -= aggregateLagrangePotentials(minimizerPermutation, lagrangePotentials, toIndex);
    return result;
  }
  // O(Tnm^2 + Nn)
  @Override
  public Pair<LinearChain, Double> findBestMinimizerResponsePermutation(
      double[] maximizerProbabilities, LinkedHashSet<LinearChain> existingMaximizerPermutations,
      double[] lagrangePotentials) {
    //Pair<Double, LinkedSparseMatrix> p0AndMatrixP =
    //    computeMarginalProbabilityMatrixP(maximizerProbabilities, existingMaximizerPermutations);
    //LinkedSparseMatrix matrixP = computeMarginalProbabilityOfTagsPerPosition(maximizerProbabilities, existingMaximizerPermutations);
	//double p0 = p0AndMatrixP.getLeft();
    //LinkedSparseMatrix matrixP = p0AndMatrixP.getRight();
    //return gfm.linearChainGFM(p0, matrixP, lagrangePotentials);
    return costSensitiveCostMaximizer(existingMaximizerPermutations, maximizerProbabilities,lagrangePotentials); 
    

/*Pair<LinearChain, Double> bestResponse = costSensitiveCostMaximizer(existingMaximizerPermutations, maximizerProbabilities,lagrangePotentials); 
    LinearChain bestResponsePermutation = bestResponse.getLeft(); 
    double bestResponseF1Value = 0;
    int permutationIndex = 0;
    double aggregateLagrangePotentialsForBestResponse = aggregateLagrangePotentials(bestResponsePermutation,lagrangePotentials); // O(n)
    for(LinearChain permutation : existingMaximizerPermutations){ //O(N)
	if (Misc.roughlyEquals(maximizerProbabilities[permutationIndex], 0)) {
				    permutationIndex++;
			            continue;
		}
	bestResponseF1Value += maximizerProbabilities[permutationIndex++] * computeScore(permutation, bestResponsePermutation);
    } 

    bestResponseF1Value -= aggregateLagrangePotentialsForBestResponse;
    //double lagrangePotentialsExpectation = computeLagrangePotentialsExpectation(
    //	        maximizerProbabilities, existingMaximizerPermutations, lagrangePotentials);
	        
    return Pair.of(bestResponsePermutation, bestResponseF1Value);*/
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
