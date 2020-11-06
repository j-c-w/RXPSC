This should just work after you run nix-shell.

Be careful to read through any of the files before you run them, e.g. bram_based_8bit.py has a hardcoded output path.

Workflow:

1. You need the software in shell.nix.  It requires some of the scripts in my j-c-w/config.

Getting ANML files.  This software works using ANML files, which
seem hard to get directly from PCRE files.  I provide a process
below from which to generate MNRL and ANML files from PCRE
files:

1. Create a PCRE (.regex) file.  This is a format of one regex
per line, e.g. '/<pattern>/'

2. Turn the pattern into a MNRL file: `pcre2mnrl <inputfile> <outputfile (.mnrl)>`

3. Turn MNRL into ANML: `vasim <mnrl file> -a`

#Running
The run script is rxpsc.py, run
that with --help to see the options.

##Modes
There are several important modes, briefly described here (see the rxpsc help for more up to date detail)

1.  Compression: Given a folder of folders, where each sub-folder contains an anml
file, try to compress the regexes assuming that all the regexes in
each folder must be run at the same time as the others in the same
folder, but that they /do not/ have to run at the same time as regexes
in different folders.
2. Addition: Given a set of regexes that have been implemented in hardware (possibly with prefix splitting), try to add another regex to that pre-compiled set
3. Various experimental modes.  These modes are designed to help recreate published (or unpublished) experiments.

#Benchmarking
Benchmarking scripts are in the temp_scripts/FCCM/benchmark_scripts/
folder.  See the local README for more information.

#On the Eddie Cluster
Eddie sadly does not support nix super well.  There is a script to
setup a partial environment --- not good enough to do the whole
setup, but good enough to run the python.  Try eddie_setup.sh
