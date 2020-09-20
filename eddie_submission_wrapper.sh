#!/bin/bash

# Get pypy
~/.scripts/EddieScripts/install_pypy.sh $TMPDIR/pypy
export PATH=$TMPDIR/pypy/bin:$PATH

cd ~/RegexSynthesis/APSim/
# Create the env
./eddie_setup.sh $TMPDIR/pyenv
source $TMPDIR/pyenv/pyenv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD

# Now, go to FCCM
cd temp_scripts/FCCM/

pypy lut_based_8bit_groups.py "$@"
echo "Job Completed!"
