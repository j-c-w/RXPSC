self: super:

let pkgs = import<nixpkgs> {};
in {
	hyperscan = super.hyperscan.overrideAttrs(old: rec {
		version = "4.4.1";
		src = super.fetchFromGitHub {
			owner = "intel";
			repo = old.pname;
			rev = "v${version}";
			sha256 = "sha256:15am0bsx92fmdlyw7byjhma4w9dcx7b7536ria7pr42nllmd7a2c";
		};

		nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.sqlite pkgs.pkg-config ];

		outputs = [ "out" "lib" "dev" ];

		# configurePhase = ''
		# 	mkdir build
		# 	cd build
		# 	cmake ..
		# 	'';

		# buildPhase = ''
		# make -j3
		# 	'';

		cmakeFlags = [
			"-DFAT_RUNTIME=ON"
			"-DBUILD_AVX512=ON"
			"-DBUILD_STATIC_AND_SHARED=ON"
		];

		postInstall = ''
			mkdir -p $lib
			cp -r lib/* $lib

			mkdir -p $dev/src
			cp -r .. $dev/src
		'';

	});
}
