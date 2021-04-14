{ pkgs ? import<nixpkgs>{ overlays = [ (import ./default.nix) ]; } }:

with pkgs;

mkShell {
	buildInputs = [ hyperscan ];
}
