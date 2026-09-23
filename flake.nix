{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    { nixpkgs, flake-parts, ... }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = nixpkgs.lib.systems.flakeExposed;
      perSystem = { pkgs, ... }: {
        devShells.default =
          let
            texlive-toolchain = pkgs.texliveSmall.withPackages (
              ps: with ps; [
                latexmk
                a4wide
                lipsum
                amsfonts
                amsmath
                xcolor
                listings
                hyperref
                cleveref
              ]
            );
            python-toolchain = pkgs.python3.withPackages (
              ps: with ps; [
                ipykernel
                jupyterlab
                jupyter
                notebook
                matplotlib
                numpy
                scipy
                pandas
                scikit-learn
              ]
            );
          in
          pkgs.mkShell {
            buildInputs = with pkgs; [
              # dotenv
              dotenvx
              # nix
              nixd
              nil
              nixfmt
              # latex
              texlive-toolchain
              texlab
              # python
              python-toolchain
              ruff
              ty
              uv
            ];

            env = {
              DOTENV_CONFIG_IGNORE = "MISSING_ENV_FILE";
            };
            shellHook = ''
              eval "$(dotenvx get --format eval-export)"
            '';
          };
      };
    };
}
