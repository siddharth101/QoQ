# QoQ (Q occupancy) project
Project with the goal of reducing the number of retractions in O4
using an analysis of Q transform pixel occupancy values

## Project organization
Project is divided into `libs`, modular source libraries for querying data, creating q transforms, calculating occupancy percentages, etc. and `projects`, that are built off the libs.

## Current projects
### injection_analysis
This project analyzes the O3 MDC replay, producing pixel occupancy values for all events with m1 and m2 greater than 5

### background_analysis
This project analyzes pycbc offline background (timeslid) triggers from O3, similarly producing pixel occupancy values for all events with m1 and m2 greater than 5


## Environment setup
To start, in the root directory, create the base Conda environment on which all projects are based. This environment should be named `QoQ-base`

```
conda env create -f environment.yaml
```

Next, install Poetry  
Poetry is the dependency manager used and can be installed via

```
curl -sSL https://install.python-poetry.org | python3 - --preview
```
Due to the fact that several tools in the Python gravitational wave analysis ecosystem can only be installed via Conda (in particular the library GWpy uses to read and write .gwf files and the library it uses for reading archival data from the NDS2 server), some projects require both Poetry and Conda.  
Projects that require Conda will have a `poetry.toml` file in them containing in part

```toml
[virtualenvs]
create = false
```

For these projects, you can build the necessary virtual environment by running

```console
conda create -n QoQ-<project-name> --clone QoQ-base
```

Then use Poetry to install additional dependencies into this environment (called from the project's root directory, not the repository's)

```console
poetry install
```

For projects that don't require conda, you should just need to run `poetry install` from the project's directory, and Poetry will take care of creating a virtual environment automatically.


## Running projects
Once the environment for the project is properly set up,
Commands can be run with custom arguments using Poetry:

```console
poetry run my-command --arg1 arg1 --arg2 arg2 
```

The available commands for projects can be seen in the `pyproject.toml`
file under the `[tool.poetry.scripts]` table.

To run the project with the default parameters specified in 
the `pyproject.toml`, run 

```console
poetry run my-command --typeo .
```

the `--typeo` flag tells poetry to look in the `pyproject.toml` file under the `[tool.typeo]` table for the default parameters. These will then be passed to the function that `my-command` is mapped to in the `pyproject.toml` file.

## Credits
This project has been developed by Sidd Soni, Ethan Marx, and Erik Katsavounidis  

Inspiration for using poetry and this project structure was taken from Alec Gunny's implementation in https://github.com/ML4GW
further resources and documentation on these tools can be found there
