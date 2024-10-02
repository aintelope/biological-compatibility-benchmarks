# Biologically and economically compatible multi-objective multi-agent AI safety benchmarks

Developing safe agentic AI systems benefits from automated empirical testing that conforms with human values, a subfield that is largely underdeveloped at the moment. To contribute towards this topic, present work focuses on introducing biologically and economically motivated themes that have been neglected in the safety aspects of modern reinforcement learning literature, namely homeostasis, balancing multiple objectives, bounded objectives, diminishing returns, sustainability, and multi-agent resource sharing. We implemented eight main benchmark environments on the above themes, for illustrating the potential shortcomings of current mainstream discussions on AI safety.

This work introduces safety challenges for an agent's ability to learn and act in desired ways in relation to biologically and economically relevant aspects. In total we implemented nine benchmarks, which are conceptually split into three developmental stages: “basic biologically inspired dynamics in objectives”, “multi-objective agents”, and “cooperation”. The first two stages can be considered as proto-cooperative stages, since the behavioral dynamics tested in these benchmarks will be later potentially very relevant for supporting and enabling cooperative behavior in multi-agent scenarios. 

The benchmarks were implemented in a gridworld-based environment. The environments are relatively simple, just as much complexity is added as is necessary to illustrate the relevant safety and performance aspects. The pictures attached in this document are illustrative, since the environment sizes and amounts of object types can be changed.

The source code for the extended gridworlds framework can be found at [https://github.com/levitation-opensource/ai-safety-gridworlds/tree/biological-compatibility-benchmarks](https://github.com/levitation-opensource/ai-safety-gridworlds/tree/biological-compatibility-benchmarks). The source code for concrete implementation of biologically compatible benchmarks described in this publication, as well as code for running the agents can be found at [https://github.com/aintelope/biological-compatibility-benchmarks](https://github.com/aintelope/biological-compatibility-benchmarks). The latter also contains example code for a random agent.

## Project setup

### Installation

The project installation is managed via `make` and `pip`. Please see the
respective commanads in the `Makefile`. To setup the environment follow these
steps:

0. Install CPython from python.org. The code is tested with Python version 3.10.10
1. Create a virtual python environment: `make venv`
2. Activate the environment: `source venv_aintelope/bin/activate`
3. Install dependencies: `make install`

For development and testing follow (active environment):

1. Install development dependencies: `make install-dev`
2. Install project locally: `make build-local`
3. Run tests: `make tests-local`

### Code formatting and style

To automatically sort the imports you can run
[`isort aintelope tests`](https://github.com/PyCQA/isort) from the root level of the project.
To autoformat python files you can use [`black .`](https://github.com/psf/black) from the root level of the project.
Configurations of the formatters can be found in `pyproject.toml`.
For linting/code style use [`flake8`](https://flake8.pycqa.org/en/latest/).

These tools can be invoked via `make`:

```bash
make isort
make format
make flake8
```

## Executing `aintelope`

Try `make run-training`. Then look in `aintelope/outputs/memory_records`. (WIP)
There should be two new files named `Record_{current timestamp}.csv` and
`Record_{current timestamp}_plot.png`. The plot will be an image of the path the
agent took during the test episode, using the best agent that the training
produced. Green dots are food in the environment, blue dots are water.

TODO

## Logging

TODO

## Windows

Aintelope code base is compatible with Windows. No extra steps needed. GPU computation works fine as well. WSL is not needed.

# Papers

* A working paper related to this repo: Pihlakas, R & Pyykkö, J. "From homeostasis to resource sharing: Biologically and economically compatible multi-objective multi-agent AI safety benchmarks". Arxiv (2024). https://arxiv.org/abs/2410.00081

# License

This project is licensed under the GNU General Public License v3.0 (GPLv3). You are free to use, modify, and distribute this code under the terms of this license.

**Attribution Requirement**: If you use this benchmark suite, please cite the source as follows:

Roland Pihlakas and Joel Pyykkö. From homeostasis to resource sharing: Biologically and economically compatible multi-objective multi-agent AI safety benchmarks. Arxiv, a working paper, September 2024.

**Use of Entire Suite**: We encourage the inclusion of the entire benchmark suite in derivative works to maintain the integrity and comprehensiveness of AI safety assessments.

For more details, see the [LICENSE.txt](LICENSE.txt) file.
