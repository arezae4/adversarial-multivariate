package edu.uic.cs.purposeful.mpg.target.linear_chain;

import java.util.Arrays;
import java.util.BitSet;
import java.util.Map;

import edu.uic.cs.purposeful.common.assertion.Assert;
import gnu.trove.map.hash.TIntObjectHashMap;
import net.mintern.primitive.pair.IntPair;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

public class LinearChain {
  private static final double EXISTENCE_VALUE = 1.0;

  private final int[] tagSequence;
  private final int length;
  private final TIntObjectHashMap<BitSet> binaryExistenceSequenceByTag;

  private final LinkedSparseMatrix unigramExistenceMatrix; // #positions * #tags
  private final LinkedSparseMatrix bigramExistenceMatrix; // #positions * #pairs

  private final int[] unigramIndices;
  private final int[] bigramIndices;

  public LinearChain(int[] tagSequence, Map<Integer, Integer> indicesByTag,
      Map<IntPair, Integer> indicesByTagPair) {
    this.tagSequence = tagSequence;
    this.length = tagSequence.length;

    this.binaryExistenceSequenceByTag = new TIntObjectHashMap<>(indicesByTag.size());
    for (int tag : indicesByTag.keySet()) {
      this.binaryExistenceSequenceByTag.put(tag, new BitSet(this.length));
    }

    this.unigramExistenceMatrix = new LinkedSparseMatrix(this.length, indicesByTag.size());
    this.bigramExistenceMatrix = new LinkedSparseMatrix(this.length, indicesByTagPair.size());
    this.unigramIndices = new int[this.length];
    this.bigramIndices = new int[this.length];

    int previousTag = LinearChainConfig.SEQUENCE_STARTING_TAG;
    for (int positionIndex = 0; positionIndex < this.length; positionIndex++) {
      int currentTag = tagSequence[positionIndex];

      BitSet binarySequence = binaryExistenceSequenceByTag.get(currentTag);
      binarySequence.set(positionIndex);

      int tagIndex = indicesByTag.get(currentTag);
      this.unigramExistenceMatrix.set(positionIndex, tagIndex, EXISTENCE_VALUE);
      this.unigramIndices[positionIndex] = tagIndex;

      if (positionIndex != 0 || LinearChainConfig.ADD_STARTING_TAG) {
        Integer tagPairIndex = indicesByTagPair.get(IntPair.of(previousTag, currentTag));
        Assert.notNull(tagPairIndex);
        this.bigramExistenceMatrix.set(positionIndex, tagPairIndex, EXISTENCE_VALUE);
        this.bigramIndices[positionIndex] = tagPairIndex;
      } else {
        this.bigramIndices[positionIndex] = -1;
      }

      previousTag = currentTag;
    }
  }

  public int getLength() {
    return length;
  }

  public int countTag(int tag) {
    return getExistenceSequence(tag).cardinality();
  }

  public int countTag(int tag, int toIndex) {
    return getExistenceSequence(tag).get(0,toIndex).cardinality();
  }

  public int countCommonTag(int tag, LinearChain that) {
    BitSet thisSequence = getExistenceSequence(tag);
    BitSet thatSequence = that.getExistenceSequence(tag);

    BitSet thisClone = (BitSet) thisSequence.clone();
    thisClone.and(thatSequence);
    return thisClone.cardinality();
  }

  public int countCommonTag(int tag, LinearChain that, int toIndex) {
    BitSet thisSequence = getExistenceSequence(tag).get(0,toIndex);
    BitSet thatSequence = that.getExistenceSequence(tag).get(0,toIndex);

    //BitSet thisClone = (BitSet) thisSequence.clone();
    thisSequence.and(thatSequence);
    return thisSequence.cardinality();
  }

  public BitSet getExistenceSequence(int tag) {
    BitSet binarySequence = binaryExistenceSequenceByTag.get(tag);
    // keySet() function has the last "}" missing...
    Assert.notNull(binarySequence,
        "Tag[" + tag + "] doesn't exist in " + binaryExistenceSequenceByTag.keySet() + "}");
    return binarySequence;
  }

  public int[] getTagSequence() {
    return tagSequence;
  }

  public LinkedSparseMatrix getUnigramExistenceMatrix() {
    return unigramExistenceMatrix;
  }

  public LinkedSparseMatrix getBigramExistenceMatrix() {
    return bigramExistenceMatrix;
  }

  public int[] getUnigramIndices() {
    return unigramIndices;
  }

  public int[] getBigramIndices() {
    return bigramIndices;
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(tagSequence);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof LinearChain)) {
      return false;
    }
    if (obj == this) {
      return true;
    }
    return Arrays.equals(tagSequence, ((LinearChain) obj).tagSequence);
  }

  @Override
  public String toString() {
    return Arrays.toString(tagSequence);
  }
}
