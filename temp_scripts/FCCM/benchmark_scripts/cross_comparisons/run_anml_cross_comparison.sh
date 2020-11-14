#!/bin/zsh

set -eu

typeset -a eddie
zparseopts -D -E -eddie=eddie

if [[ $# -ne 5 ]]; then
	echo "Usage: $0 <Input ANMLZoo Folder> <ANMLZoo Benchmark Name> <Results Folder> <flags> <runname>"
	echo "--eddie: use the submission queues on Eddie to run this"
	exit 1
fi

folder=$1/$2
results=$3
flags="$4"
runname=$5
output_name="${2}_${runname}"

input_anml_file=( $(find $folder/anml -name "*.anml" | sort) )
output_folder=$results/$output_name

if [[ $2 == Snort ]]; then
	mem=10G
else
	mem=5G
fi

if [[ ${#input_anml_file} -gt 0 ]]; then
	mkdir -p $output_folder
	# Run the tool:
	full_output_path=$(readlink -f $output_folder)
	full_anml_file=$(readlink -f ${input_anml_file[1]})
	echo $flags
	if [[ ${#eddie} -eq 0 ]]; then
		pushd ../../..
		pypy rxpsc.py "${=flags}" $full_anml_file > $full_output_path/result
	else
		# Submit to eddie.
		pushd ../../..
		qsub -o $full_output_path/result -e $full_output_path/err -pe sharedmem 2 -l h_vmem=$mem eddie_submission_wrapper.sh ${=flags} $full_anml_file
		# Add a little dely --- was getting some queueing timeouts from too many submissions I think?
		sleep 0.3
	fi
fi

echo "Run $2!"
