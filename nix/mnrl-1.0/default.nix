{ stdenvNoCC, fetchFromGitHub, gnumake, nasm, gcc6 }:

stdenvNoCC.mkDerivation {
	name = "MNRL";
	version = "1.0";
	src = fetchFromGitHub {
		owner = "kevinaangstadt";
		repo = "mnrl";
		fetchSubmodules = true;
		rev = "ea5bcdd9fabb5021f006b27ec7dee918ad2c0ddd";
		sha256 = "sha256:1v6qm8gwi2qcaihhw1rd4wya3r0v46b37rf63gxcq3niv39biyq2";
	};

	patches = [ ./noinit.patch ];

	nativeBuildInputs = [ gnumake nasm gcc6 ];

	outpts = [ "out" "lib" ];

	preBuild = ''
		cd C++
		'';

	installPhase = ''
		mkdir -p $out/src
		cp libmnrl.{so,a} $lib

		cp -r .. $out/src
		'';
}
