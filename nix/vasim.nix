{ stdenvNoCC, gcc6, fetchFromGitHub, nasm, gnumake }:

stdenvNoCC.mkDerivation {
	version = "1.0";
	name = "VASim";
	src = fetchFromGitHub {
		repo = "VASim";
		owner = "j-c-w";
		rev = "5a3c46639256a12736145363510dbf1e8c9acc7f";
		fetchSubmodules = true;
		sha256 = "sha256:0m74lr4z59arvvflxn0x1kf91i6dljil4i05n03236ir6yg407zf";
	};
	nativeBuildInputs = [ gcc6 nasm gnumake ];

	patches = [ ./noinit.patch ];

	installPhase = ''
		mkdir -p $out/bin
		mv vasim $out/bin
		'';

	# Not working for some reason.
	# doCheck = true;
	# # Don't run the official tests, because those seem broken
	# # for unknown reasons.
	# checkPhase = ''
	# 	$out/bin/vasim -h
	# 	'';
}
