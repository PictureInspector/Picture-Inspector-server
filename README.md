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

### Linux
```shell
> export FLASK_APP=/path/to/project/app.py
> export FLASK_ENV=/path/to/project/development
```

## Create Certificates to run over HTTPS
You can generate self-signed certificates easily from the command line.
All you need is to have openssl installed:
```shell
> openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```
This command writes a new certificate in cert.pem with its corresponding private key in key.pem, with a validity period of 365 days. When you run this command, you will be asked a few questions.
Here is an example of signing a certificate(instead of "server name" you should provide the name of your hosting server):
```
Generating a RSA private key
writing new private key to 'key.pem'
-----
You are about to be asked to enter information that will be incorporated
into your certificate request.
What you are about to enter is what is called a Distinguished Name or a DN.
There are quite a few fields but you can leave some blank
For some fields there will be a default value,
If you enter '.', the field will be left blank.
-----
Country Name (2 letter code) [AU]:RU
State or Province Name (full name) [Some-State]:Tatarstan Republic
Locality Name (eg, city) []:Innopolis
Organization Name (eg, company) [Internet Widgits Pty Ltd]:Innopolis University
Organizational Unit Name (eg, section) []:.
Common Name (e.g. server FQDN or YOUR name) []:<server name>
Email Address []:.
```

After creating certificate you should also copy it to the folder with the bot so that
it uses the certificate.

# Running
Type following command to run the server:
```shell script
> python3 -m flask run --host=0.0.0.0 --cert=cert.pem --key=key.pem
```