# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import os
import sys

import pytest

if __name__ == "__main__" and os.name == "nt":  # detect debugging
    pytest.main(["tests/"])  # run tests only in this folder and its subfolders
