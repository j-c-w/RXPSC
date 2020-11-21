#!/bin/zsh

set -eu
set -x

typeset -a eddie
typeset -a nosim
zparseopts -D -E -eddie=eddie -no-sim=nosim

if [[ $# -ne 3 ]]; then
	echo "Usage: $0 <Input file> <Simulators Directory> <Simulator Max Range>"
	exit 1
fi

infile=$1
simdir=$2
max_sims=$3

mkdir -p $simdir/outputs

block_size=10
mem=3G
n=0
last_n=0
simulators=( $(find $simdir -type d -wholename "*/generated_*/0" | sort) )
if [[ ${#nosim} -gt 0 ]]; then
	flags="--no-sim"
fi
# Submit by ranges.
if [[ ${#eddie} -gt 0 ]]; then
	while [[ $n -le $max_sims ]]; do
		n=$((n + block_size))

		if [[ $n -gt $max_sims ]]; then
			n=$(($3 + 1))
		fi

		qsub -P inf_regex_synthesis -o $simdir/outputs/round_std_$n -P inf_regex_synthesis -e $simdir/outputs/round_err_$n -l h_vmem=$mem run_simulation.sh $PWD $infile $simdir ${simulators[@]:$last_n:$n} --eddie $flags
		sleep 0.1

		last_n=$n
	done
else
	set -x
	parallel ./run_simulation.sh $PWD $infile $simdir {} $flags > $simdir/outputs/{n}.out ::: ${simulators[@]}
fi
