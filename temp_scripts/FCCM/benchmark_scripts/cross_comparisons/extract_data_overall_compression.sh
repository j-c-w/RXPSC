#!/bin/bash
set -x
set -eu

if [[ $# -ne 2 ]]; then
	echo "Usage: $0 <results folder> <output file>"
fi

result=$1
of=$2

for file in $(find $result -wholename "$result/*___allow_overapproximation___no_groups___compression_stats/result"); do
	# We have the file: get the benchmark name:
	bname=$(echo $(basename $(dirname $file)) | cut -d'_' -f1)

	if [[ $(grep -c 'COMPILATION STATISTICS:' $file) -eq 0 ]]; then
		continue
	fi

	# Now, get the base and the reduced numbers
	self_compiles=$(grep -e 'COMPILATION STATISTICS: self compiles ='  $file | cut -d'=' -f2)
	regex_reduction=$(grep -e 'COMPILATION STATISTICS: reduction in regexes =' $file | cut -d'=' -f2)
	if (( self_compiles == 0 )); then
		# Happens for benchmarks with regexes that are too large
		continue
	fi

	echo "$bname $(( regex_reduction / (self_compiles + regex_reduction)))" >> $of
done
