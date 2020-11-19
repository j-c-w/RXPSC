#!/bin/zsh

set -eu

typeset -a eddie
zparseopts -D -E -eddie=eddie

if [[ $# -ne 2 ]]; then
	echo "Usage: $0 <Groups Location> <Results Folder>"
	echo "--eddie: use the submission queues on Eddie to run this"
	exit 1
fi

# Used to run shorter experiments, should be just very very high by default.
files_to_execute=200

benchmark=$1
results=$2

other_flags="--compression-stats --no-structural-change --allow-overapproximation --backend python"

flag_combinations=(
	"--use-prefix-splitting --prefix-size-threshold 1 --use-prefix-estimation"
	"--use-prefix-splitting --prefix-size-threshold 1 --use-prefix-estimation --prefix-merging-only --no-prefix-unification"
	# ""
	)

flag_combination_names=(
	"unification"
	"nounification"
)

files_to_run=(  $(find $benchmark -name "*.anml" ) )

findex=1
for file in ${files_to_run[@]}; do
	if [[ $findex -gt $files_to_execute ]]; then
		break
	fi
	fcomb_index=1
	for fcomb in ${flag_combinations[@]}; do
		# The tool filters out the file from being used
		# in the to /and/ from set, so we don't have to worry
		# about doing anything about that.
		echo $file
		echo $fcomb
		name=${flag_combination_names[fcomb_index]}
		of_flag="--output-folder $results/sim/$name/$(basename $file)"
		if [[ ${#eddie} -gt 0 ]]; then
			./snort/run_for_file.sh $name $file $benchmark $results "addition-experiment $fcomb $other_flags $of_flag" --eddie
			sleep 0.1 # Submitting jobs too fast is bad for eddie.
		else
			./snort/run_for_file.sh $name $file $benchmark $results "addition-experiment $fcomb $other_flags $of_flag"
		fi
		fcomb_index=$((fcomb_index + 1))
	done
	findex=$((findex + 1))
done
