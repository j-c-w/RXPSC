#!/bin/zsh

set -eu

typeset -a eddie
zparseopts -D -E -eddie=eddie

if [[ $# -ne 5 ]]; then
	echo "Usage: $0 <name> <To Accelerate File> <Accelerated Files> <Output Folder> <Flags>"
	echo "--eddie: use the submission queues on Eddie to run this"
	exit 1
fi
mem=7G

name=$1
to_accelerate=$2
accelerated=$3
results=$4
flags="$5"

output_folder=$results/${name}_${to_accelerate_name}

mkdir -p $output_folder

full_output_path=$(readlink -f $output_folder)
full_accelerated_path=$(readlink -f ${accelerated})
full_to_accelerate_path=$(readlink -f ${to_accelerate})
to_accelerate_name=$(basename $to_accelerate)

if [[ ${#eddie} -eq 0 ]]; then
	cd ../../..
	pypy rxpsc.py "${=flags}" $full_to_accelerate_path $full_accelerated_path > $full_output_path/result
else
	cd ../../..
	qsub -o $full_output_path/result_$to_accelerate_name -e $full_output_path/err_$to_accelerate_name -pe sharedmem 2 -l h_vmem=$mem eddie_submission_wrapper.sh ${=flags} $full_to_accelerate_path $full_accelerated_path
fi
