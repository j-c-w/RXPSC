# { pkgs, callPackage, stdenv, fetchFromGitHub, boost, hyperscan, cmake }:

# This is obviously a dirty hack --- not sure how to get this into the args
# while still passing the overlays.
with import<nixpkgs> { overlays = [ (import ./hyperscan-source/default.nix  ) ]; };

let
	mnrl_package = (callPackage ./mnrl-1.0/default.nix {});
	mnrl_deriv = (import ./mnrl-1.0/default.nix );

	# overlayPkgs = pkgs.override {
	# 	overlays = [  ];
	# };
in
let
	mnrl_outpath = mnrl_package.outPath;
in
stdenv.mkDerivation {
	name = "HSCompile";

	src = fetchFromGitHub {
		owner = "kevinaangstadt";
		repo = "hscompile";
		rev = "a032a0afb733a6e6912db04c2237f3ee50f053bd";
		sha256 = "sha256:01r7vl1y9lp685f52nxzws1mgzskzgq9665wr1iibpyqh4wwz2w0";
	};

	# patches = [ ./libfinding.patch ];

	nativeBuildInputs = [ cmake
		boost
		(callPackage ./mnrl-1.0/default.nix {})
		hyperscan
	];

	cmakeFlags = [
		"-DHS_SOURCE_DIR=${pkgs.hyperscan.dev}/src"
		"-DHS_BUILD_DIR=${pkgs.hyperscan.dev}/src/build"
		"-DMNRL_SOURCE_DIR=${mnrl_outpath}/src/C++"
	];

	installPhase = ''
		mkdir -p $out/bin
		cp hscompile hsrun pcre2mnrl $out/bin
		'';
}
