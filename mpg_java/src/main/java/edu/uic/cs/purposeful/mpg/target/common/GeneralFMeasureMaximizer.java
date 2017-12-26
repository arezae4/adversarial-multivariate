package edu.uic.cs.purposeful.mpg.target.common;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;

import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.target.linear_chain.LinearChain;
import edu.uic.cs.purposeful.mpg.target.linear_chain.LinearChainConfig;
import edu.uic.cs.purposeful.mpg.target.linear_chain.f1.LinearChainF1;
import gnu.trove.list.array.TIntArrayList;
import net.mintern.primitive.pair.DoubleIntPair;
import net.mintern.primitive.pair.IntPair;
import net.mintern.primitive.pair.ObjDoublePair;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.UpperSymmDenseMatrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

public class GeneralFMeasureMaximizer {
  private static class ComputeMatrixW implements Function<Integer, UpperSymmDenseMatrix> {
    @Override
    public UpperSymmDenseMatrix apply(Integer totalNumOfPositions) {
      outputToConsole("Begin computing matrixW for " + totalNumOfPositions);
      UpperSymmDenseMatrix w = new UpperSymmDenseMatrix(totalNumOfPositions);
      for (int rowIndex = 0; rowIndex < totalNumOfPositions; rowIndex++) {
        for (int columnIndex = rowIndex; columnIndex < totalNumOfPositions; columnIndex++) {
          w.set(rowIndex, columnIndex, 1.0 / (rowIndex + columnIndex + 2));
        }
      }
      outputToConsole("End computing matrixW for " + totalNumOfPositions);
      return w;
    }
  }

  private static final ConcurrentHashMap<Integer, UpperSymmDenseMatrix> MATRIX_W_CACHE =
      new ConcurrentHashMap<>();

  private static final TIntArrayList EMPTY_INT_LIST = new TIntArrayList(0);

  private final int targetTag;
  private final Map<Integer, Integer> indicesByTag;
  private final Map<IntPair, Integer> indicesByTagPair;

  protected final int numOfPositions;

  protected static void outputToConsole(String info) {
    if (MPGConfig.SHOW_RUNNING_TRACING) {
      System.err.println(info);
    }
  }

  public GeneralFMeasureMaximizer(int targetTag, Map<Integer, Integer> indicesByTag,
      Map<IntPair, Integer> indicesByTagPair, int totalNumOfPositions) {
    this.numOfPositions = totalNumOfPositions;
    this.targetTag = targetTag;
    this.indicesByTag = indicesByTag;
    this.indicesByTagPair = indicesByTagPair;
  }

  public GeneralFMeasureMaximizer(int totalNumOfPositions) {
    this.numOfPositions = totalNumOfPositions;

    // useless fields for #gfm(double, LinkedSparseMatrix)
    this.targetTag = Integer.MIN_VALUE;
    this.indicesByTag = null;
    this.indicesByTagPair = null;
  }

  protected UpperSymmDenseMatrix getMatrixW() {
    return MATRIX_W_CACHE.computeIfAbsent(numOfPositions, new ComputeMatrixW());
  }

  /**
   * This method returns a permutation that <b>MAXIMIZE</b> the F1 score;
   */
  public Pair<BitSet, Double> gfm(double p0, LinkedSparseMatrix matrixP) {
    return gfm(p0, matrixP, null);
  }

  /**
   * If lagrangePotentials <b>is null</b>, this method returns a permutation that <b>MAXIMIZE</b>
   * the F1 score; otherwise, it returns a permutation <b>MINIMIZE</b> the F1 score.
   */
  public Pair<BitSet, Double> gfm(double p0, LinkedSparseMatrix matrixP,
      double[] lagrangePotentials) {
    boolean findMaximize = lagrangePotentials == null;
    Assert.isTrue(findMaximize || lagrangePotentials.length == numOfPositions);

    DenseMatrix scoreMatrix = null;
    if (findMaximize) {
      scoreMatrix = new DenseMatrix(numOfPositions, numOfPositions);
      matrixP.mult(2, getMatrixW(), scoreMatrix);
    } else {
      scoreMatrix = initializeScoreMatrixWithNegativeLagrangePotentials(lagrangePotentials);
      matrixP.multAdd(2, getMatrixW(), scoreMatrix);
    }

    List<List<DoubleIntPair>> rowIndicesByOrderedValueInColumns =
        storeAndSortPositionIndices(scoreMatrix, findMaximize);

    return findTheBestResponse(p0, rowIndicesByOrderedValueInColumns, findMaximize);
  }

  protected List<List<DoubleIntPair>> storeAndSortPositionIndices(DenseMatrix scoreMatrix,
      boolean findMaximize) {
    List<List<DoubleIntPair>> rowIndicesByOrderedValueInColumns =
        new ArrayList<>(scoreMatrix.numColumns());
    for (int index = 0; index < scoreMatrix.numColumns(); index++) {
      rowIndicesByOrderedValueInColumns.add(new ArrayList<>(scoreMatrix.numRows()));
    }

    for (MatrixEntry entry : scoreMatrix) {
      List<DoubleIntPair> indicesByOrderedValue =
          rowIndicesByOrderedValueInColumns.get(entry.column());
      indicesByOrderedValue.add(DoubleIntPair.of(entry.get(), entry.row()));
    }

    for (List<DoubleIntPair> indicesByOrderedValue : rowIndicesByOrderedValueInColumns) {
      if (findMaximize) {
        Collections.sort(indicesByOrderedValue, Collections.reverseOrder());
      } else {
        Collections.sort(indicesByOrderedValue);
      }
    }

    return rowIndicesByOrderedValueInColumns;
  }

  protected Pair<BitSet, Double> findTheBestResponse(double p0,
      List<List<DoubleIntPair>> rowIndicesByOrderedValueInColumns, boolean findMaximize) {
    double bestValueSum = p0;
    int bestPermutationNumOfOnes = 0;
    BitSet bestPermutation = new BitSet(numOfPositions); // all zeros

    for (int columnIndex = 0; columnIndex < rowIndicesByOrderedValueInColumns
        .size(); columnIndex++) {
      List<DoubleIntPair> indicesByOrderedValue =
          rowIndicesByOrderedValueInColumns.get(columnIndex);
      int numOfOnes = columnIndex + 1;
      BitSet permutation = new BitSet(numOfPositions);
      double valueSum = 0.0;

      for (int retriveIndex = 0; retriveIndex < numOfOnes; retriveIndex++) {
        DoubleIntPair indexByValue = indicesByOrderedValue.get(retriveIndex);
        valueSum += indexByValue.getLeft();
        permutation.set(indexByValue.getRight());
      }

      if ((findMaximize && valueSum > bestValueSum) || (!findMaximize && valueSum < bestValueSum)) {
        bestValueSum = valueSum;
        bestPermutationNumOfOnes = numOfOnes;
        bestPermutation = permutation;
      }
    }

    // If using sparse matrix it is possible the best permutation has only several high/low order
    // bits which has non-zero values; we still need to fill more bits to reach the total number of
    // ones, even those bits have zero values.
    // int oneBitsNeedToFill = bestPermutationNumOfOnes - bestPermutation.cardinality();
    // Assert.isTrue(oneBitsNeedToFill >= 0);
    // if (oneBitsNeedToFill > 0) {
    // BitSet oneBitsAvailable = new BitSet(totalNumOfPositions);
    // oneBitsAvailable.set(0, totalNumOfPositions);
    // oneBitsAvailable.xor(bestPermutation); // find bit positions haven't been used
    //
    // int count = 0;
    // for (int bitIndex = oneBitsAvailable.nextSetBit(0); count < oneBitsNeedToFill
    // && bitIndex >= 0; bitIndex = oneBitsAvailable.nextSetBit(bitIndex + 1), count++) {
    // bestPermutation.set(bitIndex);
    // }
    // }

    Assert.isTrue(bestPermutation.cardinality() == bestPermutationNumOfOnes);
    return Pair.of(bestPermutation, bestValueSum);
  }

  private DenseMatrix initializeScoreMatrixWithNegativeLagrangePotentials(
      double[] lagrangePotentials) {
    double[] negativeLagrangePotentials = new double[lagrangePotentials.length];
    for (int index = 0; index < lagrangePotentials.length; index++) {
      negativeLagrangePotentials[index] = -lagrangePotentials[index];
    }
    double[] negativeLagrangePotentialMatrixInArray = new double[numOfPositions * numOfPositions];
    for (int index = 0; index < numOfPositions; index++) {
      System.arraycopy(negativeLagrangePotentials, 0, negativeLagrangePotentialMatrixInArray,
          index * numOfPositions, numOfPositions);
    }
    return new DenseMatrix(numOfPositions, numOfPositions, negativeLagrangePotentialMatrixInArray,
        false);
  }

  public Pair<LinearChain, Double> linearChainGFM(double p0, LinkedSparseMatrix matrixP,
      double[] lagrangePotentials) {
    LinkedSparseMatrix scoreMatrix = new LinkedSparseMatrix(numOfPositions, numOfPositions);
    matrixP.mult(2, getMatrixW(), scoreMatrix);

    double bestSum = Double.NEGATIVE_INFINITY;
    TIntArrayList bestTags = null;

    for (int targetTagTotalCount =
        0; targetTagTotalCount <= numOfPositions; targetTagTotalCount++) {
      HashMap<Triple<Integer, Integer, Integer>, ObjDoublePair<TIntArrayList>> cache =
          new HashMap<>();
      ObjDoublePair<TIntArrayList> bestForTheTagTotalCount =
          maxsum(LinearChainConfig.SEQUENCE_STARTING_TAG, 0, targetTagTotalCount,
              targetTagTotalCount, scoreMatrix, lagrangePotentials, cache);

      TIntArrayList bestTagsForTheTagTotalCount = bestForTheTagTotalCount.getLeft();
      double bestSumForTheTagTotalCount = bestForTheTagTotalCount.getRight();

      // F1 of all zeros vs all zeros is 1 instead of 0 (with probability p0)
      if (targetTagTotalCount == 0) {
        Assert.isTrue(
            new LinearChain(bestTagsForTheTagTotalCount.toArray(), indicesByTag, indicesByTagPair)
                .countTag(targetTag) == 0);
        bestSumForTheTagTotalCount -= p0;
      }

      if (bestSumForTheTagTotalCount > bestSum) {
        bestSum = bestSumForTheTagTotalCount;
        bestTags = bestTagsForTheTagTotalCount;
      }
    }

    return Pair.of(new LinearChain(bestTags.toArray(), indicesByTag, indicesByTagPair), -bestSum);
  }

  private ObjDoublePair<TIntArrayList> maxsum(int previousTag, int currentPositionIndex,
      int targetTagRemainingCount, int targetTagTotalCount, LinkedSparseMatrix scoreMatrix,
      double[] lagrangePotentials,
      Map<Triple<Integer, Integer, Integer>, ObjDoublePair<TIntArrayList>> cache) {
    Triple<Integer, Integer, Integer> cacheKey =
        Triple.of(previousTag, currentPositionIndex, targetTagRemainingCount);
    ObjDoublePair<TIntArrayList> cachedBest = cache.get(cacheKey);
    if (cachedBest != null) {
      return cachedBest;
    }

    if (currentPositionIndex >= numOfPositions) { // i >= end
      double bestSum;
      if (targetTagRemainingCount > 0) {
        bestSum = Double.NEGATIVE_INFINITY;
      } else {
        bestSum = 0;
      }
      ObjDoublePair<TIntArrayList> best = ObjDoublePair.of(EMPTY_INT_LIST, bestSum);
      cache.put(cacheKey, best);
      return best;
    }

    // score at a certain position
    double score;
    if (targetTagTotalCount == 0) {
      score = 0;
    } else {
      // be careful! index in scoreMatrix = targetTagTotalCount - 1
      score = scoreMatrix.get(currentPositionIndex, targetTagTotalCount - 1);
    }

    double bestSum = Double.NEGATIVE_INFINITY;
    TIntArrayList bestTags = null;

    for (int currentTag : indicesByTag.keySet()) {
      boolean isCurrentTagTarget = (currentTag == targetTag);
      if (targetTagRemainingCount <= 0 && isCurrentTagTarget) {
        continue; // impossible since there should be no target tag any more
      }

      ObjDoublePair<TIntArrayList> nextBest = maxsum(currentTag, currentPositionIndex + 1,
          isCurrentTagTarget ? (targetTagRemainingCount - 1) : targetTagRemainingCount,
          targetTagTotalCount, scoreMatrix, lagrangePotentials, cache);

      double nextMaxSum = nextBest.getRight();
      if (Double.isInfinite(nextMaxSum)) { // can't be the best
        Assert.isTrue(nextMaxSum < 0, "Must be Double.NEGATIVE_INFINITY");
        continue;
      }

      double potential = retrieveLagrangePotential(previousTag, currentTag, currentPositionIndex,
          lagrangePotentials);
      double currentSum = potential - (isCurrentTagTarget ? score : 0.0) + nextMaxSum;

      if (currentSum > bestSum) {
        bestSum = currentSum;
        TIntArrayList nextBestTags = nextBest.getLeft();
        bestTags = new TIntArrayList(nextBestTags.size() + 1);
        bestTags.add(currentTag);
        bestTags.addAll(nextBestTags);
      }
    }

    ObjDoublePair<TIntArrayList> best = ObjDoublePair.of(bestTags, bestSum);
    cache.put(cacheKey, best);
    return best;
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
}
