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
# Can also be eg. compress
mode=addition-experiment-anml-zoo

# bmarks=( Snort Protomata )
# bmarks=(  Brill Bro217 ClamAV Custom Dotstar Dotstar03 Dotstar06 Dotstar09 EntityResolution ExactMath Fermi Hamming Levenshtein PowerEN Protomata RandomForest Ranges05 Ranges1 Snort SPM Synthetic TCP )
# This is the list that actually work/exist:
bmarks=( Brill ClamAV Dotstar PowerEN Protomata Snort Synthetic )
optional_flags=( --allow-overapproximation --no-structural-change "--target symbol-only-reconfiguration" )
other_flags="--allow-overapproximation --compression-stats --no-structural-change --backend python"

flag_combinations=(
	# "--use-prefix-merging"
	# "--cross-compile --allow-overapproximation"
	# "--cross-compile --use-prefix-merging --allow-overapproximation"
	# "--cross-compile --target symbol-only-reconfiguration --no-structural-change"
	# "--cross-compile"
	# "--cross-compile --use-prefix-merging"
	# ""
	# "--allow-overapproximation"
	# "--use-prefix-splitting --use-prefix-estimation --prefix-merging-only --no-prefix-unification"
	"--use-prefix-splitting --use-prefix-estimation --output-folder $results/sim/nounification/"
	# "--use-prefix-splitting --use-prefix-estimation --prefix-merging-only --no-prefix-unification --prefix-size-threshold 2"
	# "--use-prefix-splitting --use-prefix-estimation"
	# "--use-prefix-splitting --use-prefix-estimation --prefix-size-threshold 2"
	# ""
	# "--use-prefix-splitting"
	# "--no-structural-change"
	# "--target symbol-only-reconfiguration"
	# "--allow-overapproximation --no-structural-change"
	# "--allow-overapproxmation --target symbol-only-reconfiguration"
	# "--no-structural-change --target symbol-only-reconfiguration"
	# "--allow-overapproximation --no-structural-change --target symbol-only-reconfiguration"
	)

runno=0
for bmark in ${bmarks[@]}; do
	for fcomb in $flag_combinations[@];
	if [[ ${#eddie} -gt 0 ]]; then
		./cross_comparisons/run_anml_cross_comparison.sh $ANMLZoo $bmark $results "$mode $fcomb $other_flags --output-folder $results/sim/$bmark" $runno --eddie
	else
		./cross_comparisons/run_anml_cross_comparison.sh $ANMLZoo $bmark $results "$mode $fcomb $other_flags --output-folder $results/sim/$bmark" $runno
	fi
	runno=$((runno + 1))
done
