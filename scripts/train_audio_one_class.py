from __future__ import annotations

import sys

from train_mimii_one_class import main


if __name__ == "__main__":
    main(["--dataset-format", "folders", *sys.argv[1:]])
