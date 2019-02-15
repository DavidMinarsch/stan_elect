# Stan Elect

An environment to run a Stan example.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

This assumes you manage python versions with `pyenv` as outlined [here](https://github.com/pyenv/pyenv) and virtual environments with `pyenv-virtualenv` as outlined [here](https://github.com/pyenv/pyenv-virtualenv).

See `.python-version` for the python version required.

See `requirements.txt` for the libraries required.

### Setup Virtual Environment:

Manage python versions with `pyenv`. Check python versions installed include version specified in 
`.python-version` (i.e. :
```
$ pyenv versions
```

Manage virtual environments with `pyenv-virtualenvs`. Create a virtual environment:
```
$ pyenv virtualenv 3.7.1 stan_elect
```

### Install Dependencies:

CD into folder to activate the virtual environment and run:
```
$ pip install requirements.txt
```

## Run Notebook
Install a new kernel:
```
$ ipython kernel install --user --name=stan_elect
```
Start the notebook server:
```
$ jupyter notebook
```
Run the notebook from the browser!

## Or run from Command Line
```
$ python <filename>
```
