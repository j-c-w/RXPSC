#!/bin/bash

set -eu

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <results>"
	exit 1
fi

for file in $1/*; do
	outname=$(basename $file)
	if [[ ! -d $file/outputs ]]; then
		continue
	fi
	pushd $file/outputs
	rg 'Overacceptances ' --no-filename | cut -f 2 -d' ' > ../../$outname.overacceptance.dat
	rg 'Average Accept Length ' --no-filename | cut -f 4 -d' ' > ../../$outname.accepting_lenths.dat
	popd
done
