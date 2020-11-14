#!/bin/bash

set -eu

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <results folder>"
	exit 1
fi

folder=$1
cd $folder

failed_count=0
converted_count=0
for file in $(find -name "result*" | sort); do
	converted=$(rg "COMPRESSION RESULT: Converted" -c $file | cut -f2 -d':')
	failed=$(rg "COMPRESSION RESULT: Failed" -c $file | cut -f2 -d':')

	failed_count=$((failed_count + failed))
	converted_count=$((converted_count + converted))
done

echo "Failed is $failed_count"
echo "Converted is $converted_count"
