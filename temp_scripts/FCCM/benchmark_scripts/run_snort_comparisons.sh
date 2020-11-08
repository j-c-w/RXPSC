#!/bin/zsh

set -eu

typeset -a eddie
zparseopts -D -E -eddie=eddie

if [[ $# -ne 2 ]]; then
	echo "Usage: $0 <Groups Location> <Results Folder>"
	echo "--eddie: use the submission queues on Eddie to run this"
	exit 1
fi

benchmark=$1
results=$2

other_flags="--compression-stats --no-structural-change --allow-overapproximation"

flag_combinations=(
	"--use-prefix-splitting --use-prefix-estimation"
	"--use-prefix-splitting --use-prefix-estimation --prefix-merging-only --no-prefix-unification"
	""
	)

files_to_run=(  $(find $benchmark -name "*.anml" ) )

for file in ${files_to_run[@]}; do
	for fcomb in ${flag_combinations[@]}; do
		# The tool filters out the file from being used
		# in the to /and/ from set, so we don't have to worry
		# about doing anything about that.
		echo $file
		echo $fcomb
		if [[ ${#eddie} -gt 0 ]]; then
			./snort/run_for_file.sh $file $benchmark "addition-experiment $fcomb $other_flags" $results --eddie
		else
			./snort/run_for_file.sh $file $benchmark "addition-experiment $fcomb $other_flags" $results
		fi
	done
done
