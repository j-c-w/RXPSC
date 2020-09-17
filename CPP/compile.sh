#!/bin/bash
set -eu

echo "Compiling"
Python_Path="$(dirname $(which python))/../include/python2.7"
Pypy_Path="$(dirname $(which pypy))/../pypy-c/include/"
echo $Python_Path

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac

if [ ! -d $Python_Path ]
then
    echo "Python include path does not exist"
    exit 1 # die with error code 1
fi

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 [pypy|cpython]"
	exit 1
fi

rm -f *.so *.o *.cxx *.pyc *.py
touch __init__.py
g++ -c -fPIC -fPIC -std=c++11  VASim.cpp
swig -c++ -python VASim.i
if [[ $1 == cpython ]]; then
	echo Using $Python_Path
	g++ -c -fPIC -std=c++11 VASim_wrap.cxx  -I $Python_Path
else
	echo Using $Pypy_Path
	g++ -c -fPIC -std=c++11 VASim_wrap.cxx  -I $Pypy_Path
fi

if [ $machine = Linux ]
then
    g++ -shared -Wl,-soname,_VASim.so -o _VASim.so VASim.o VASim_wrap.o
elif [ $machine = Mac ]
then
    g++ -undefined dynamic_lookup  VASim.o VASim_wrap.o -o _VASim.so
else
    echo "this operating system is not supported"
    exit 2    
fi

me=`dirname "$0"`
so_path="$me/_VASim.so"

if [ -f "$so_path" ]
then
	echo "Done!"
else
	echo "Unsuccessful. Please follow instructions in www.swig.org to install swig and run this script again"
	exit 1
fi
