package edu.uic.cs.purposeful.mpg.learning.linear_chain;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.commons.math3.util.MathUtils;

import com.carrotsearch.sizeof.RamUsageEstimator;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Iterables;
import com.google.common.primitives.Ints;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import edu.uic.cs.purposeful.common.assertion.Assert;
import edu.uic.cs.purposeful.common.assertion.PurposefulBaseException;
import edu.uic.cs.purposeful.common.collection.CollectionUtils;
import edu.uic.cs.purposeful.mpg.MPGConfig;
import edu.uic.cs.purposeful.mpg.common.Misc;
import edu.uic.cs.purposeful.mpg.target.linear_chain.LinearChainConfig;
import net.mintern.primitive.pair.IntPair;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;

public class LinearChainDataSet {
  public static enum FeatureType {
    U, // unigram
    B; // bigram
  }

  public static class LinearChainDataSetInstance {
    private final int qid;
    private final int[] goldenTags;

    private final CompRowMatrix unigramFeatureMatrix;
    private final CompRowMatrix bigramFeatureMatrix;
    private final CompRowMatrix unigramTransposedFeatureMatrix;
    private final CompRowMatrix bigramTransposedFeatureMatrix;

    private final Map<Integer, Integer> indicesByTag;
    private final Map<IntPair, Integer> indicesByTagPair;

    public LinearChainDataSetInstance(int qid, int[] goldenTags, Map<Integer, Integer> indicesByTag,
        Map<IntPair, Integer> indicesByTagPair, CompRowMatrix unigramFeatureMatrix,
        CompRowMatrix bigramFeatureMatrix, CompRowMatrix unigramTransposedFeatureMatrix,
        CompRowMatrix bigramTransposedFeatureMatrix) {
      Assert.isTrue(goldenTags.length == unigramFeatureMatrix.numRows());
      Assert.isTrue(goldenTags.length == bigramFeatureMatrix.numRows());
      Assert.isTrue(goldenTags.length == unigramTransposedFeatureMatrix.numColumns());
      Assert.isTrue(goldenTags.length == bigramTransposedFeatureMatrix.numColumns());
      this.qid = qid;
      this.goldenTags = goldenTags;
      this.indicesByTag = indicesByTag;
      this.indicesByTagPair = indicesByTagPair;
      this.unigramFeatureMatrix = unigramFeatureMatrix;
      this.bigramFeatureMatrix = bigramFeatureMatrix;
      this.unigramTransposedFeatureMatrix = unigramTransposedFeatureMatrix;
      this.bigramTransposedFeatureMatrix = bigramTransposedFeatureMatrix;
    }

    public Map<Integer, Integer> getIndicesByTag() {
      return Collections.unmodifiableMap(indicesByTag);
    }

    public Map<IntPair, Integer> getIndicesByTagPair() {
      return Collections.unmodifiableMap(indicesByTagPair);
    }

    public int[] getGoldenTags() {
      return goldenTags;
    }

    public int getQid() {
      return qid;
    }

    public CompRowMatrix getUnigramFeatureMatrix() {
      return unigramFeatureMatrix;
    }

    public CompRowMatrix getBigramFeatureMatrix() {
      return bigramFeatureMatrix;
    }

    public CompRowMatrix getUnigramTransposedFeatureMatrix() {
      return unigramTransposedFeatureMatrix;
    }

    public CompRowMatrix getBigramTransposedFeatureMatrix() {
      return bigramTransposedFeatureMatrix;
    }

    // @Override
    // public String toString() {
    // List<TreeMap<Integer, String>> rows = new ArrayList<>(goldenTags.length);
    // for (int rowIndex = 0; rowIndex < goldenTags.length; rowIndex++) {
    // rows.add(new TreeMap<>());
    // }
    //
    // fillFeatureStrings(unigramFeatureMatrix, originalIndicesByUnigramFeatureIndex, FeatureType.U,
    // rows);
    // fillFeatureStrings(bigramFeatureMatrix, originalIndicesByBigramFeatureIndex, FeatureType.B,
    // rows);
    //
    // StringBuilder toString = new StringBuilder();
    // for (int rowIndex = 0; rowIndex < rows.size(); rowIndex++) {
    // toString.append(goldenTags[rowIndex]);
    // toString.append(" qid:").append(qid);
    //
    // for (String featureString : rows.get(rowIndex).values()) {
    // toString.append(featureString);
    // }
    //
    // if (rowIndex < goldenTags.length - 1) {
    // toString.append(IOUtils.LINE_SEPARATOR);
    // }
    // }
    // return toString.toString();
    // }
    //
    // private void fillFeatureStrings(LinkedSparseMatrix matrix,
    // TIntIntMap originalIndicesByFeatureIndex, FeatureType featureType,
    // List<TreeMap<Integer, String>> rows) {
    // for (MatrixEntry entry : matrix) {
    // int featureIndex = entry.column();
    // int originalFeatureIndex = originalIndicesByFeatureIndex.get(featureIndex);
    // Assert.isTrue(originalFeatureIndex >= 0);
    // // index in the original file is 1-based
    // String featureString = new StringBuilder(" ").append(originalFeatureIndex + 1)
    // .append(featureType).append(":").append(entry.get()).toString();
    // rows.get(entry.row()).put(originalFeatureIndex, featureString);
    // }
    // }
  }

  private static final String QID = "qid";

  private final int numOfUnigramFeatures;
  private final int numOfBigramFeatures;
  private final Map<Integer, Integer> indicesByTag;
  private final Map<IntPair, Integer> indicesByTagPair;
  private final List<LinearChainDataSetInstance> instances;

  private LinearChainDataSet(int numOfUnigramFeatures, int numOfBigramFeatures,
      Map<Integer, Integer> indicesByTag, Map<IntPair, Integer> indicesByTagPair,
      List<LinearChainDataSetInstance> instances) {
    this.numOfUnigramFeatures = numOfUnigramFeatures;
    this.numOfBigramFeatures = numOfBigramFeatures;
    this.indicesByTag = indicesByTag;
    this.indicesByTagPair = indicesByTagPair;
    this.instances = instances;

    if (MPGConfig.SHOW_RUNNING_TRACING) {
      System.err.printf(
          "DataSet >>>> numOfInstance=%d, indicesByTag=%s, indicesByTagPair=%s, numOfUnigramFeatures=%d, numOfBigramFeatures=%d\n",
          instances.size(), indicesByTag, indicesByTagPair, numOfUnigramFeatures,
          numOfBigramFeatures);
    }
  }

  public List<LinearChainDataSetInstance> getInstances() {
    return instances;
  }

  public int getNumOfUnigramFeatures() {
    return numOfUnigramFeatures;
  }

  public int getNumOfBigramFeatures() {
    return numOfBigramFeatures;
  }

  public Map<Integer, Integer> getIndicesByTag() {
    return indicesByTag;
  }

  public Map<IntPair, Integer> getIndicesByTagPair() {
    return indicesByTagPair;
  }

  public int getNumOfTags() {
    return indicesByTag.size();
  }

  public int getNumOfTagPairs() {
    return indicesByTagPair.size();
  }

  public static LinearChainDataSet binarize(LinearChainDataSet originalDataSet, int targetTag) {
    if (MPGConfig.SHOW_RUNNING_TRACING) {
      System.err.println("Binarizing data set for tag [" + targetTag + "]...");
    }

    Assert.isTrue(originalDataSet.indicesByTag.containsKey(targetTag),
        "Original effective tags should contain target tag [" + targetTag + "].");
    Assert.isFalse(originalDataSet.indicesByTag.containsKey(-targetTag),
        "Original effective tags should not contain [-" + targetTag
            + "], which will be used in the binarized data set.");

    TreeSet<Integer> effectiveTags = new TreeSet<>(Arrays.asList(targetTag, -targetTag));
    TreeMap<Integer, Integer> indicesByTag = initializeTagIndices(effectiveTags);
    Map<IntPair, Integer> indicesByTagPair = initializeTagPairIndices(effectiveTags);

    List<LinearChainDataSetInstance> originalInstances = originalDataSet.instances;
    List<LinearChainDataSetInstance> instances = new ArrayList<>(originalInstances.size());
    for (LinearChainDataSetInstance originalInstance : originalInstances) {
      int[] originalGoldenTags = originalInstance.goldenTags;
      int[] goldenTags = new int[originalGoldenTags.length];
      for (int index = 0; index < originalGoldenTags.length; index++) {
        goldenTags[index] = (originalGoldenTags[index] == targetTag) ? targetTag : -targetTag;
      }
      LinearChainDataSetInstance instance = new LinearChainDataSetInstance(originalInstance.qid,
          goldenTags, indicesByTag, indicesByTagPair, originalInstance.unigramFeatureMatrix,
          originalInstance.bigramFeatureMatrix, originalInstance.unigramTransposedFeatureMatrix,
          originalInstance.bigramTransposedFeatureMatrix);
      instances.add(instance);
    }

    LinearChainDataSet dataSet = new LinearChainDataSet(originalDataSet.numOfUnigramFeatures,
        originalDataSet.numOfBigramFeatures, indicesByTag, indicesByTagPair, instances);

    if (MPGConfig.SHOW_RUNNING_TRACING) {
      System.err.println("Binarized data set for tag [" + targetTag + "]");
    }
    return dataSet;
  }

  public static LinearChainDataSet loadFromFile(File file) {
    List<String> lines;
    try {
      lines = FileUtils.readLines(file);
    } catch (IOException e) {
      throw new PurposefulBaseException(e);
    }

    TreeSet<Integer> effectiveTags = new TreeSet<>();
    TreeMap<Integer, ArrayList<Integer>> tagsByQid = new TreeMap<>();
    TreeMap<Integer, ArrayList<List<Feature>>> unigramFeaturessByQid = new TreeMap<>();
    TreeMap<Integer, ArrayList<List<Feature>>> bigramFeaturessByQid = new TreeMap<>();

    int[] numsOfFeatures = parseLines(lines, effectiveTags, tagsByQid, unigramFeaturessByQid,
        bigramFeaturessByQid, null, false);
    int numOfUnigramFeatures = numsOfFeatures[0];
    int numOfBigramFeatures = numsOfFeatures[1];
    // check #instances matches
    Assert.isTrue(tagsByQid.size() == unigramFeaturessByQid.size());
    Assert.isTrue(tagsByQid.size() == bigramFeaturessByQid.size());

    if (MPGConfig.BIAS_FEATURE_VALUE >= 0) {
      numOfUnigramFeatures++; // put the bias feature among unigram features
    }

    TreeMap<Integer, Integer> indicesByTag = initializeTagIndices(effectiveTags);
    Map<IntPair, Integer> indicesByTagPair = initializeTagPairIndices(effectiveTags);

    List<LinearChainDataSetInstance> instances = new ArrayList<>(tagsByQid.size());
    Set<Integer> qids = tagsByQid.keySet();
    int firstQid = Iterables.getFirst(qids, null);
    for (Integer qid : qids) {
      List<Integer> tags = tagsByQid.get(qid);
      List<List<Feature>> unigramFeaturess = unigramFeaturessByQid.get(qid);
      List<List<Feature>> bigramFeaturess = bigramFeaturessByQid.get(qid);
      // check #positions matches
      Assert.isTrue(tags.size() == unigramFeaturess.size());
      Assert.isTrue(tags.size() == bigramFeaturess.size());

      CompRowMatrix[] unigramFeatureMatrices =
          buildFeatureMatrix(unigramFeaturess, numOfUnigramFeatures, MPGConfig.BIAS_FEATURE_VALUE);
      // just put the bias feature among unigram features
      CompRowMatrix[] bigramFeatureMatrices =
          buildFeatureMatrix(bigramFeaturess, numOfBigramFeatures, null);

      LinearChainDataSetInstance instance = new LinearChainDataSetInstance(qid, Ints.toArray(tags),
          indicesByTag, indicesByTagPair, unigramFeatureMatrices[0], bigramFeatureMatrices[0],
          unigramFeatureMatrices[1], bigramFeatureMatrices[1]);
      instances.add(instance);
      if (MPGConfig.SHOW_RUNNING_TRACING && (qid == firstQid || qid % 100 == 0)) {
        System.err.printf("Instance qid=[%d]/%d is loaded...\n", qid, tagsByQid.size());
      }
    }

    LinearChainDataSet dataSet = new LinearChainDataSet(numOfUnigramFeatures, numOfBigramFeatures,
        indicesByTag, indicesByTagPair, instances);
    if (MPGConfig.SHOW_RUNNING_TRACING) {
      System.err.printf("DataSet >>>> [%s], size=%s\n", file.getPath(),
          Misc.byteCountToHumanReadableSize(RamUsageEstimator.sizeOf(dataSet), true));
    }
    return dataSet;
  }

  private static CompRowMatrix[] buildFeatureMatrix(List<List<Feature>> featuress,
      int numOfFeatures, Double biasFeatureValue) {
    LinkedSparseMatrix _featureMatrix = new LinkedSparseMatrix(featuress.size(), numOfFeatures);
    LinkedSparseMatrix _transposedFeatureMatrix =
        new LinkedSparseMatrix(numOfFeatures, featuress.size());
    for (int rowIndex = 0; rowIndex < featuress.size(); rowIndex++) {
      List<Feature> features = featuress.get(rowIndex);
      for (Feature feature : features) {
        _featureMatrix.set(rowIndex, feature.getIndex(), feature.getValue());
        _transposedFeatureMatrix.set(feature.getIndex(), rowIndex, feature.getValue());
      }

      if (biasFeatureValue != null && biasFeatureValue.doubleValue() >= 0) { // bias feature
        _featureMatrix.set(rowIndex, numOfFeatures - 1, biasFeatureValue);
        _transposedFeatureMatrix.set(numOfFeatures - 1, rowIndex, biasFeatureValue);
      }
    }
    CompRowMatrix featureMatrix = new CompRowMatrix(_featureMatrix);
    CompRowMatrix transposedFeatureMatrix = new CompRowMatrix(_transposedFeatureMatrix);
    return new CompRowMatrix[] {featureMatrix, transposedFeatureMatrix};
  }

  private static int[] parseLines(List<String> lines, TreeSet<Integer> effectiveTags,
      TreeMap<Integer, ArrayList<Integer>> tagsByQid,
      TreeMap<Integer, ArrayList<List<Feature>>> unigramFeaturessByQid,
      TreeMap<Integer, ArrayList<List<Feature>>> bigramFeaturessByQid,
      TreeMap<Integer, ArrayList<List<Feature>>> allFeaturessByQid, boolean recordGlobal) {

    // all 1-based, initialized with 0
    MutableInt maxUnigramFeatureIndex = new MutableInt();
    MutableInt maxBigramFeatureIndex = new MutableInt();
    MutableInt maxAllFeatureIndex = new MutableInt();

    for (String line : lines) {
      line = StringUtils.substringBefore(line, "#").trim();
      if (line.isEmpty()) {
        continue;
      }

      String[] parts = StringUtils.split(line);
      Assert.isTrue(parts.length >= 3, "At lease one feature.");

      String[] qidParts = StringUtils.split(parts[1], ":");
      Assert.isTrue(qidParts.length == 2);
      Assert.isTrue(QID.equals(qidParts[0]));
      Integer qid = Integer.valueOf(qidParts[1]);

      Integer tag = Integer.valueOf(parts[0]);
      effectiveTags.add(tag);
      CollectionUtils.putInArrayListValueMap(qid, tag, tagsByQid);

      List<Feature> unigramFeatures = new ArrayList<>();
      List<Feature> bigramFeatures = new ArrayList<>();
      List<Feature> allFeatures = new ArrayList<>();

      for (int partIndex = 2; partIndex < parts.length; partIndex++) {
        String[] featureParts = StringUtils.split(parts[partIndex], ":");
        Assert.isTrue(featureParts.length == 4);

        MutableInt maxFeatureIndex = null;
        List<Feature> features = null;
        int featureIndex;

        if (recordGlobal) {
          maxFeatureIndex = maxAllFeatureIndex;
          features = allFeatures;
          featureIndex = Integer.parseInt(featureParts[0]);
        } else {
          FeatureType featureType = FeatureType.valueOf(featureParts[1]);
          if (featureType == FeatureType.U) {
            maxFeatureIndex = maxUnigramFeatureIndex;
            features = unigramFeatures;
          } else if (featureType == FeatureType.B) {
            maxFeatureIndex = maxBigramFeatureIndex;
            features = bigramFeatures;
          } else {
            Assert.isTrue(false);
          }
          featureIndex = Integer.parseInt(featureParts[2]);
        }

        Assert.isTrue(featureIndex >= 1); // 1-based
        maxFeatureIndex.setValue(Math.max(featureIndex, maxFeatureIndex.intValue()));
        featureIndex--; // to 0-based

        double featureValue = Double.parseDouble(featureParts[3]);
        if (MathUtils.equals(featureValue, 0)) {
          continue;
        }
        features.add(new FeatureNode(featureIndex, featureValue));
      }

      if (recordGlobal) {
        Assert.isTrue(unigramFeatures.isEmpty());
        Assert.isTrue(bigramFeatures.isEmpty());
        CollectionUtils.putInArrayListValueMap(qid, allFeatures, allFeaturessByQid);
      } else {
        Assert.isTrue(allFeatures.isEmpty());
        CollectionUtils.putInArrayListValueMap(qid, unigramFeatures, unigramFeaturessByQid);
        CollectionUtils.putInArrayListValueMap(qid, bigramFeatures, bigramFeaturessByQid);
      }
    }

    if (recordGlobal) {
      Assert.isTrue(maxUnigramFeatureIndex.intValue() == 0);
      Assert.isTrue(maxBigramFeatureIndex.intValue() == 0);
      return new int[] {maxAllFeatureIndex.intValue()};
    } else {
      Assert.isTrue(maxAllFeatureIndex.intValue() == 0);
      return new int[] {maxUnigramFeatureIndex.intValue(), maxBigramFeatureIndex.intValue()};
    }
  }

  private static TreeMap<Integer, Integer> initializeTagIndices(Set<Integer> effectiveTags) {
    TreeMap<Integer, Integer> indicesByTag = new TreeMap<>();
    for (Integer effectiveTag : effectiveTags) {
      indicesByTag.put(effectiveTag, indicesByTag.size());
    }
    return indicesByTag;
  }

  @VisibleForTesting
  static Map<IntPair, Integer> initializeTagPairIndices(Set<Integer> effectiveTags) {
    Map<IntPair, Integer> indicesByTagPair = new HashMap<>();
    if (LinearChainConfig.ADD_STARTING_TAG) {
      for (int effectiveTag : effectiveTags) {
        Assert.isFalse(effectiveTags.contains(LinearChainConfig.SEQUENCE_STARTING_TAG),
            String.format(
                "Tags%s used in data set cannot contain [%d], "
                    + "change the value of 'sequence_starting_tag' in 'mpg_linear_chain_config.properties' to another one.",
                effectiveTags, LinearChainConfig.SEQUENCE_STARTING_TAG));

        IntPair startTagPair = IntPair.of(LinearChainConfig.SEQUENCE_STARTING_TAG, effectiveTag);
        indicesByTagPair.put(startTagPair, indicesByTagPair.size());
      }
    }

    for (int effectiveTag1 : effectiveTags) {
      for (int effectiveTag2 : effectiveTags) {
        IntPair tagPair = IntPair.of(effectiveTag1, effectiveTag2);
        indicesByTagPair.put(tagPair, indicesByTagPair.size());
      }
    }
    return indicesByTagPair;
  }
}
