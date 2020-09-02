#!/bin/bash

# Make a python environment if it doesn't exist:
if [[ ! -d pyenv/ ]]; then
	virtualenv -p python pyenv
fi

if [[ ! -f CPP/_VASim.so ]]; then
	cd CPP
	./compile.sh
	cd ..
fi

source ./pyenv/bin/activate
pip install enum sortedcontainers numpy matplotlib pathos networkx deap tqdm jinja2
