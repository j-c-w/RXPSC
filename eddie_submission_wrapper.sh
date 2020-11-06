#!/bin/bash

# Get pypy
~/.scripts/EddieScripts/install_pypy.sh $TMPDIR/pypy
export PATH=$TMPDIR/pypy/bin:$PATH

cd ~/RegexSynthesis/APSim/
# Create the env
./eddie_setup.sh $TMPDIR/pyenv
source $TMPDIR/pyenv/pyenv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD

pypy rxpsc.py "$@"
echo "Job Completed!"
