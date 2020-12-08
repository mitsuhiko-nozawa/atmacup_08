import yaml
import argparse
import sys, os
sys.path.append("../../libs")
import warnings
warnings.filterwarnings("ignore")

from run import Run
import pandas as pd
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    CONFIG = yaml.safe_load(open(args.config))

    run = Run(CONFIG)
    run()


if __name__ == "__main__":
    main()