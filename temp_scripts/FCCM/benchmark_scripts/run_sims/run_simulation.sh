#!/bin/zsh
set -eu

typeset -a eddie
typeset -a nosim
zparseopts -D -E -eddie=eddie -no-sim=nosim

if [[ $# -lt 3 ]]; then
	echo "Usage $0 <This directory (for eddie, but used elsewhere)> <Input File> <Simulators Files>"
	exit 1
fi

if [[ ${#eddie} -gt 0 ]]; then
	# Install pypy, some of this processing is really long...:
	~/.scripts/EddieScripts/install_pypy.sh $TMPDIR/pypy
	export PATH=$PATH:$TMPDIR/pypy/bin
fi

set -x

cd $1
shift

input_file=$(readlink -f $1)
sim_file=$2
TIMEOUT_TIME=300 # 5 minutes.  Should be incresed for bigger data

shift 2
while [[ $# -gt 0 ]]; do
	generated=$1
	orig=${1/generated_/original_}
	shift

	# Run both the inputs
	if [[ ${#nosim} -eq 0 ]]; then
		pushd $generated
		for file in $(find -name "*.py"); do
			timeout $TIMEOUT_TIME pypy $file 0 $input_file $file.report || echo "FAILED DUE TO TIME" > $file.report
		done
		popd

		pushd $orig
		for file in $(find -name "*.py"); do
			timeout $TIMEOUT_TIME pypy $file 0 $input_file $file.report || echo "FAILED DUE TO TIME" > $file.report
		done
		popd
	fi

	# Now, put together all the report files to find overall
	# acceptances.
	pypy check_acceptances.py $orig/*.report $generated/*.report 
done
