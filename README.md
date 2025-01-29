# Biologically and economically aligned multi-objective multi-agent AI safety benchmarks

Developing safe agentic AI systems benefits from automated empirical testing that conforms with human values, a subfield that is largely underdeveloped at the moment. To contribute towards this topic, present work focuses on introducing biologically and economically motivated themes that have been neglected in the safety aspects of modern reinforcement learning literature, namely homeostasis, balancing multiple objectives, bounded objectives, diminishing returns, sustainability, and multi-agent resource sharing. We implemented eight main benchmark environments on the above themes, for illustrating the potential shortcomings of current mainstream discussions on AI safety.

This work introduces safety challenges for an agent's ability to learn and act in desired ways in relation to biologically and economically relevant aspects. In total we implemented nine benchmarks, which are conceptually split into three developmental stages: “basic biologically inspired dynamics in objectives”, “multi-objective agents”, and “cooperation”. The first two stages can be considered as proto-cooperative stages, since the behavioral dynamics tested in these benchmarks will be later potentially very relevant for supporting and enabling cooperative behavior in multi-agent scenarios. 

The benchmarks were implemented in a gridworld-based environment. The environments are relatively simple, just as much complexity is added as is necessary to illustrate the relevant safety and performance aspects. The pictures attached in this document are illustrative, since the environment sizes and amounts of object types can be changed.

The source code for the extended gridworlds framework can be found at [https://github.com/levitation-opensource/ai-safety-gridworlds/tree/biological-compatibility-benchmarks](https://github.com/levitation-opensource/ai-safety-gridworlds/tree/biological-compatibility-benchmarks). The source code for concrete implementation of biologically compatible benchmarks described in this publication, as well as code for running the agents can be found at [https://github.com/aintelope/biological-compatibility-benchmarks](https://github.com/aintelope/biological-compatibility-benchmarks). The latter also contains example code for a random agent.

## Project setup

This readme contains instructions for both Linux and Windows installation. Windows installation instructions are located after Linux installation instructions.

### Installation under Linux

The project installation is managed via `make` and `pip`. Please see the respective commands in the `Makefile`. To setup the environment follow these steps:

1. Install CPython. The code is tested with Python version 3.10.10. We do not recommend using Conda package manager. 

Under Linux, run the following commands:

`sudo add-apt-repository ppa:deadsnakes/ppa`
<br>`sudo apt update`
<br>`sudo apt install python3.10 python3.10-dev python3.10-venv`
<br>`sudo apt install curl`
<br>`sudo curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10`

2. Get the code from repo:

`sudo apt install git-all`
<br>Run `git clone https://github.com/aintelope/biological-compatibility-benchmarks.git`
<br>Run `cd biological-compatibility-benchmarks`

3. Create a virtual python environment:

`make venv-310`
<br>`source venv_aintelope/bin/activate`

4. Install dependencies:

`sudo apt update`
<br>`sudo apt install build-essential`
<br>`make install`

5. If you use VSCode, then set up your launch configurations file:

`cp .vscode/launch.json.template .vscode/launch.json`

Edit the launch.json so that the PYTHONPATH variable points to the folder where you downloaded the repo and installed virtual environment:

replace all
<br>//"PYTHONPATH": "your_path_here"
<br>with
<br>"PYTHONPATH": "your_local_repo_path"

6. For development and testing:

* Install development dependencies: `make install-dev`
* Run tests: `make tests-local`

7. Location of an example agent you can use as a template for building your custom agent: 
[`aintelope/agents/example_agent.py`](aintelope/agents/example_agent.py)


### Installation under Windows

1. Install CPython from python.org. The code is tested with Python version 3.10.10. We do not recommend using Conda package manager.

You can download the latest installer from https://www.python.org/downloads/release/python-31010/ or if you want to download a newer 3.10.x version then from https://github.com/adang1345/PythonWindows

2. Get the code from repo:
* Install Git from https://gitforwindows.org/
* Open command prompt and navigate top the folder you want to use for repo
* Run `git clone https://github.com/aintelope/biological-compatibility-benchmarks.git`
* Run `cd biological-compatibility-benchmarks`

3. Create a virtual python environment by running: 
<br>3.1. To activate VirtualEnv with Python 3.10:
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`virtualenv -p python3.10 venv_aintelope` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(or if you want to use your default Python version: 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`python -m venv venv_aintelope`)
<br>3.2. `venv_aintelope\scripts\activate`

4. Install dependencies by running:
<br>`pip uninstall -y ai_safety_gridworlds >nul 2>&1`
<br>`pip install -r requirements/api.txt`

5. If you use VSCode, then set up your launch configurations file:

`copy .vscode\launch.json.template .vscode\launch.json`

Edit the launch.json so that the PYTHONPATH variable points to the folder where you downloaded the repo and installed virtual environment:

replace all
<br>//"PYTHONPATH": "your_path_here"
<br>with
<br>"PYTHONPATH": "your_local_repo_path"

6. For development and testing:

* Install development dependencies: `pip install -r requirements/dev.txt`
* Run tests: `python -m pytest --tb=native --cov="aintelope tests"`

7. Location of an example agent you can use as a template for building your custom agent: 
[`aintelope\agents\example_agent.py`](aintelope/agents/example_agent.py)


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

In the folder `.vscode` there is a file named `launch.json.template`. Copy that file to `launch.json`. This is a VSCode launch configurations file, containing many launch configurations. (The original file named `launch.json.template` is necessary so that your local changes to launch configurations do not end up in the Git repository.)

Alternatively, try executing `make run-training-baseline`. You do not need VSCode for running this command. Then look in `aintelope/outputs`. This command will execute only one of many available launch configurations present in `launch.json`.

### Executing LLM agent

For LLM agent, there are the following launch configurations in `launch.json`:
- Run single environment with LLM agent and default params
- Run pipeline with LLM agent and default params
- Run BioBlue pipeline with LLM agent and default params
- Run multiple trials pipeline with LLM agent and default params
- Run multiple trials BioBlue pipeline with LLM agent and default params


## Logging

TODO

## Windows

Aintelope code base is compatible with Windows. No extra steps needed. GPU computation works fine as well. WSL is not needed.

# Papers

* A working paper related to this repo: Pihlakas, R & Pyykkö, J. "From homeostasis to resource sharing: Biologically and economically compatible multi-objective multi-agent AI safety benchmarks". Arxiv (2024). https://arxiv.org/abs/2410.00081

# Blog posts

* Why modelling multi-objective homeostasis is essential for AI alignment (and how it helps with AI safety as well) (2025) https://www.lesswrong.com/posts/vGeuBKQ7nzPnn5f7A/why-modelling-multi-objective-homeostasis-is-essential-for

# Presentations

* At VAISU unconference, May 2024:
    - Demo and feedback session - AI safety benchmarking in multi-objective multi-agent gridworlds - Biologically essential yet neglected themes illustrating the weaknesses and dangers of current industry standard approaches to reinforcement learning. 
    - Video: https://www.youtube.com/watch?v=ydxMlGlQeco
    - Slides: https://bit.ly/bmmbs
* At Foresight Institute's Intelligent Cooperation Group, Nov 2024: 
    - The subject of the presentation was describing why we should consider fundamental yet neglected principles from biology and economics when thinking about AI alignment, and how these considerations will help with AI safety as well (alignment and safety were treated in this research explicitly as separate aspects, which both benefit from consideration of aforementioned principles). These principles include homeostasis and diminishing returns in utility functions, and sustainability. Next I will introduce multi-objective and multi-agent gridworlds-based benchmark environments we have created for measuring the performance of machine learning algorithms and AI agents in relation to their capacity for biological and economical alignment. The benchmarks are now available as a public repo. At the end I will mention some of the related themes and dilemmas not yet covered by these benchmarks, and describe new benchmark environments we have planned for future implementation.
    - Recording: https://www.youtube.com/watch?v=DCUqqyyhcko
    - Slides: https://bit.ly/beamm 

# License

This project is licensed under the Mozilla Public License 2.0. You are free to use, modify, and distribute this code under the terms of this license.

**Attribution Requirement**: If you use this benchmark suite, please cite the source as follows:

Roland Pihlakas and Joel Pyykkö. From homeostasis to resource sharing: Biologically and economically compatible multi-objective multi-agent AI safety benchmarks. Arxiv, a working paper, September 2024 (https://arxiv.org/abs/2410.00081).

**Use of Entire Suite**: We encourage the inclusion of the entire benchmark suite in derivative works to maintain the integrity and comprehensiveness of AI safety assessments.

For more details, see the [LICENSE.txt](LICENSE.txt) file.
