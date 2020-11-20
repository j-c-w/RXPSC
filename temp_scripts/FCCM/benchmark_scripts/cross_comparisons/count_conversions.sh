#!/bin/zsh

set -eu

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <results folder>"
	exit 1
fi

folder=$1
cd $folder

for file in $(find -name "result" | sort); do
	echo $file:
	echo "Failed: $(rg "COMPRESSION RESULT: Failed" -c $file)"
	echo "Converted: $(rg "COMPRESSION RESULT: Converted" -c $file)"

	time_numbers=( $(rg "TIMING: Time taken" $file | cut -f5 -d' ') )
	tot=0
	count=0
	for n in ${time_numbers[@]};do
		tot=$((tot + n))
		count=$((count + 1))
	done

	echo "Time taken (mean): $(( tot / count ))"
done
