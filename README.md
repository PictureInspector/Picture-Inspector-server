# Picture-Inspector-server

# Installation

## Python installation
To run this server you need to install [Python3+](https://realpython.com/installing-python/).
And [pip](https://pip.pypa.io/en/stable/installation/).

## Create virtual environment
You need to create new virtual environment. Type following commands:
```shell script
> python3 -m venv env_name
> source env_name/bin/activate
```

## Install libraries and weights
Install all needed libraries using this command:
```shell script
> pip install -r /path/to/project/requirements.txt
```
Install the weights for neural network and dataset using this [link](https://drive.google.com/file/d/1mj239x6k7s1S5kljo-3hoyXEro5kKfRE/view?usp=sharing). And extract it to the following path:
```shell script
~/app/data/
```

## Set environment variables
Type following commands to set environment variables
### Windows
```shell
> set FLASK_APP=/path/to/project/app.py
> set FLASK_ENV=/path/to/project/development
```

###Linux
```shell
> export FLASK_APP=/path/to/project/app.py
> export FLASK_ENV=/path/to/project/development
```

# Running
Type following command to run the server:
```shell script
> python3 -m flask run --host=0.0.0.0
```