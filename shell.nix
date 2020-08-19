{ pkgs ? import <nixpkgs> {} }:


with pkgs;
let
	pythonenv = pkgs.python27.buildEnv.override {
		extraLibs = with pkgs.python27Packages; [ 
			enum
			sortedcontainers
			numpy
			matplotlib
			pathos
			networkx
			deap
			tqdm
			jinja2
			pygraphviz
			repoze_lru
		];
		ignoreCollisions = true;
	};

	# Get HSCompile
	hscompile = (import ~/.scripts/Nix/CustomPackages/AutomataTools/hscompile/default.nix );
in
pkgs.mkShell {
	SHELL_NAME = "APSim";
	buildInputs = [ pythonenv swig
	# Get VASim tools
	(callPackage ~/.scripts/Nix/CustomPackages/AutomataTools/vasim/default.nix {})  
	verilog
	hscompile
	ctags
	];

	shellHook = ''
		if [[ ! -f CPP/_VASim.so ]]; then
			cd CPP
			# Build whatever thing the README told us to.
			./compile.sh
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
