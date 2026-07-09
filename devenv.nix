{ pkgs, ... }:

{
  # Rust toolchain — stable, matching CI. There is no rustfmt.toml, so stable
  # rustfmt output is identical to CI's nightly rustfmt job.
  languages.rust = {
    enable = true;
    channel = "stable";
    components = [
      "rustc"
      "cargo"
      "clippy"
      "rustfmt"
      "rust-analyzer"
    ];
  };

  # `cargo audit` — the RUSTSEC advisory gate CI runs.
  packages = [ pkgs.cargo-audit ];

  # `devenv test` mirrors .github/workflows/ci.yml (check + test + fmt + clippy).
  enterTest = ''
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace
  '';

  # See the full reference at https://devenv.sh/reference/options/
}
