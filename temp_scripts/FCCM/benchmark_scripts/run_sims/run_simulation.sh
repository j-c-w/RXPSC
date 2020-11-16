#!/bin/bash
set -eu

if [[ $# -lt 2 ]]; then
	echo "Usage $0 <Input File> <Simulators File> <Range (numbers)>"
	exit 1
fi

cd $(dirname $0) # Eddie starts this in the top directory.

input_file=$(readlink -f $1)
sim_file=$2

shift 2
while [[ $# -gt 0 ]]; do
	generated=$sim_file/generated_$1
	orig=$sim_file/original_$1
	shift

	# Run both the inputs
	pushd $generated/0
	for file in $(find -name "*.py"); do
		python2 $file 0 $input_file $file.report
	done
	popd

	pushd $orig/0
	for file in $(find -name "*.py"); do
		python2 $file 0 $input_file $file.report
	done
	popd

	# Now, put together all the report files to find overall
	# acceptances.
	python2 check_acceptances.py $orig/0/*.report $generated/0/*.report 
done
