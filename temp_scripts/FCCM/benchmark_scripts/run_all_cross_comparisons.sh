#!/bin/zsh

typeset -a eddie
zparseopts -D -E -eddie=eddie

if [[ $# -ne 2 ]]; then
	echo "Usage: $0 <ANMLZoo Location> <Results Location>"
	echo "--eddie: use the submission queues on Eddie to run this"
	exit 1
fi

ANMLZoo=$1
results=$2

bmarks=( Brill Bro217 ClamAV Custom Dotstar Dotstar03 Dotstar06 Dotstar09 EntityResolution ExactMath Fermi Hamming Levenshtein PowerEN Protomata RandomForest Ranges05 Ranges1 Snort SPM Synthetic TCP )
optional_flags=( --allow-overapproximation --no-structural-change "--target symbol-only-reconfiguration" )
other_flags="--no-groups --compression-stats"

flag_combinations=(
	""
	"--allow-overapproximation"
	"--no-structural-change"
	"--target symbol-only-reconfiguration"
	"--allow-overapproximation --no-structural-change"
	"--allow-overapproxmation --target symbol-only-reconfiguration"
	"--no-structural-change --target symbol-only-reconfiguration"
	"--allow-overapproximation --no-structural-change --target symbol-only-reconfiguration"
	)

for bmark in ${bmarks[@]}; do
	for fcomb in $flag_combinations[@];
	if [[ ${#eddie} -gt 0 ]]; then
		./cross_comparisons/run_anml_cross_comparison.sh $ANMLZoo $bmark $results "$fcomb $other_flags" --eddie
	else
		./cross_comparisons/run_anml_cross_comparison.sh $ANMLZoo $bmark $results "$fcomb $other_flags"
	fi
done
