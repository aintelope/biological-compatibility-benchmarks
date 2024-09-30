import os
import sys

import pytest

if __name__ == "__main__" and os.name == "nt":  # detect debugging
    pytest.main(["tests/"])  # run tests only in this folder and its subfolders
