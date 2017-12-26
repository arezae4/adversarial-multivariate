package edu.uic.cs.purposeful.mpg.target.linear_chain;

import edu.uic.cs.purposeful.common.config.AbstractConfig;

public class LinearChainConfig extends AbstractConfig {
  private LinearChainConfig() {
    super("mpg_linear_chain_config.properties");
  }

  private static final LinearChainConfig INSTANCE = new LinearChainConfig();

  public static final boolean ADD_STARTING_TAG = INSTANCE.getBooleanValue("add_starting_tag");
  public static final int SEQUENCE_STARTING_TAG = INSTANCE.getIntValue("sequence_starting_tag");
}
