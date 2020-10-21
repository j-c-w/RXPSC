#!/bin/zsh

set -eu

typeset -a eddie
zparseopts -D -E -eddie=eddie

if [[ $# -ne 4 ]]; then
	echo "Usage: $0 <Input ANMLZoo Folder> <ANMLZoo Benchmark Name> <Results Folder> <flags>"
	echo "--eddie: use the submission queues on Eddie to run this"
	exit 1
fi

folder=$1/$2
results=$3
flags="$4"
output_name="${2}_$(echo $flags | tr -- '- ' '_')"

input_anml_file=( $(find $folder/anml -name "*.anml" | sort) )
output_folder=$results/$output_name

if [[ ${#input_anml_file} -gt 0 ]]; then
	mkdir -p $output_folder
	# Run the tool:
	full_output_path=$(readlink -f $output_folder)
	full_anml_file=$(readlink -f ${input_anml_file[1]})
	echo $flags
	if [[ ${#eddie} -eq 0 ]]; then
		pushd ..
		pypy lut_based_8bit_groups.py -f test $full_anml_file "${=flags}" > $full_output_path/result
	else
		# Submit to eddie.
		pushd ../..
		qsub -o $full_output_path/result -e $full_output_path/err -pe sharedmem 2 -l h_vmem=5G eddie_submission_wrapper.sh -f test $full_anml_file ${=flags}
	fi
fi

echo "Run $2!"
