#!/bin/bash

cd ~/RegexSynthesis/APSim/
source pyenv/bin/activate

export PYTHONPATH=$PYTHONPATH:$PWD

# Now, go to FCCM
cd temp_scripts/FCCM/

python lut_based_8bit_groups.py "$@"
echo "Job Completed!"
