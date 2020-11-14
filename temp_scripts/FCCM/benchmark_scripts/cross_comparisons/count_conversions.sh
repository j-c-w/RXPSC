#!/bin/bash

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <results folder>"
fi

folder=$1
cd $folder

for file in $(find -name "result" | sort); do
	echo $file:
	echo "Failed: $(rg "COMPRESSION RESULT: Failed" -c $file)"
	echo "Converted: $(rg "COMPRESSION RESULT: Converted" -c $file)"
done
