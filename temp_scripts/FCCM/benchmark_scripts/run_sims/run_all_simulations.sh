#!/bin/zsh

set -eu

typeset -a eddie
zparseopts -D -E -eddie=eddie

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
# Submit by ranges.
if [[ ${#eddie} -gt 0 ]]; then
	while [[ $n -le $max_sims ]]; do
		n=$((n + block_size))

		if [[ $n -gt $max_sims ]]; then
			n=$(($3 + 1))
		fi

		set -x
		qsub -o $simdir/outputs/round_std_$n -P inf_regex_synthesis -e $simdir/outputs/round_err_$n -l h_vmem=$mem run_simulation.sh $PWD $infile $simdir $(seq  -s ' ' $last_n $((n - 1)))
		sleep 0.3

		last_n=$n
	done
else
	set -x
	parallel ./run_simulation.sh $PWD $infile $simdir {} ::: $(seq -s' ' 0 $max_sims)
fi
