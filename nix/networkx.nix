{ pkgs, fetchFromGitHub, buildPythonPackage, pythonPkgs }:

buildPythonPackage {
    pname = "networkx";
    version = "2.2";
    doCheck = false;
    src = fetchFromGitHub {
        owner = "networkx";
        repo = "networkx";
        rev = "networkx-2.2";
        sha256 = "sha256:1ivhpffxniy3f7qksphv7a0hwdycyadqc5yyikl9d82yvg24j0mh";
    };

    propagatedBuildInputs = with pythonPkgs; [
        decorator nose
    ];
}
