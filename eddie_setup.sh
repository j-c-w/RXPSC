#!/bin/bash
if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <location>"
fi

mkdir -p $1

cd $1

# Clear the pip env if it exists
if [[ -d pyenv/ ]]; then
	rm -rf pyenv
fi

virtualenv -p pypy pyenv
cd -

if [[ ! -f CPP/_VASim.so ]]; then
	cd CPP
	./compile.sh
	cd ..
fi

cd $1
source ./pyenv/bin/activate
pip install enum sortedcontainers numpy pathos networkx deap tqdm jinja2 enum34
cd -
