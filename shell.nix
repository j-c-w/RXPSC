{ pkgs ? import <nixpkgs> {} }:


with pkgs;
let 
	use_pypy = true;
	pythonEnvs = rec {
		pkgsList = ppkgs: with ppkgs; [
		enum tqdm sortedcontainers numpy pathos
		pygraphviz jinja2
		];
		# pypy doesn't support alll the packages
		# particularly well.
		cpythonPkgsList = ppkgs: with ppkgs; [
			memory_profiler
			deap
			matplotlib 
			setuptools
		];
		# pypy also needs some extra stuff
		pypyPkgsList = ppkgs: with ppkgs; [
			setuptools
		];
		pypyenv = pkgs.pypy2.buildEnv.override {
			extraLibs = pkgsList pkgs.pypy2.pkgs ++ pypyPkgsList pkgs.pypyPackages;
			ignoreCollisions = true;
		};
		cpythonenv = pkgs.python27.buildEnv.override {
			extraLibs = pkgsList pkgs.python27Packages ++ cpythonPkgsList pkgs.python27Packages;
			ignoreCollisions = true;
		};
		pythonenv = if use_pypy then pypyenv else cpythonenv;
};

	pythonpkgs = if use_pypy then pypy.pkgs else python2.pkgs;
	# Get HSCompile
	hscompile = (import ~/.scripts/Nix/CustomPackages/AutomataTools/hscompile/default.nix );
in
pkgs.mkShell {
	SHELL_NAME = "APSim";
	buildInputs = [ pythonEnvs.pythonenv swig
	# networkx
	(callPackage ~/.scripts/Nix/CustomPackages/PythonTools/networkx/default.nix {buildPythonPackage = pythonpkgs.buildPythonPackage; pythonPkgs = pythonpkgs; pkgs = pkgs;} )
	# deap
	(callPackage ~/.scripts/Nix/CustomPackages/PythonTools/deap/default.nix {buildPythonPackage = pythonpkgs.buildPythonPackage; pythonPkgs = pythonpkgs; pkgs = pkgs;} )
	# Get VASim tools
	(callPackage ~/.scripts/Nix/CustomPackages/AutomataTools/vasim/default.nix {})  
	# (callPackage ~/.scripts/Nix/CustomPackages/PythonTools/Guppy/default.nix {buildPythonPackage = pythonpkgs.buildPythonPackage; pythonPkgs = pythonpkgs; pkgs = pkgs;} )
	# Guppy only supports cpython, pypy isn't supported.
	# (callPackage ~/.scripts/Nix/CustomPackages/PythonTools/Guppy/default.nix {buildPythonPackage = pypy2.pkgs.buildPythonPackage; ppythonPkgs = pypy2.pkgs; pkgs = pkgs; })
	# verilog
	hscompile
	ctags
	];

	shellHook = ''
		if [[ ! -f CPP/_VASim.so ]]; then
			cd CPP
			# Build whatever thing the README told us to.
			./compile.sh pypy
			cd ..
		else
		  echo "It seems that _VASim.so is already built, skipping"
		fi

		if [[ ! -d ANMLZoo ]]; then
		  git clone https://github.com/gr-rahimi/ANMLZoo
		else
		  echo "Already have ANMLZoo"
	    fi

		echo "Replacing the _base_address variable in anml_zoo.py"
		sed -i "s|_base_address = .*$|_base_address = '$PWD/ANMLZoo'|" automata/AnmalZoo/anml_zoo.py

		# Need this regardless.
		export PYTHONPATH=$PYTHONPATH:$PWD
	'';
}
