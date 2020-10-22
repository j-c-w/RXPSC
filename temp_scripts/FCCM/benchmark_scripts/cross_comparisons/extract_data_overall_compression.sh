#!/bin/bash

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <results folder> <output folder>"
fi

result=$1
of=$2

for file in $result/*__allow_overapproximation__no_groups__compression_stats/result; do
	echo $file
done
