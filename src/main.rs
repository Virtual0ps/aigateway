//! `aigateway` — CLI entry point for the loopback Anthropic → OpenAI-compatible
//! gateway sidecar.
//!
//! Usage:
//!
//! ```text
//! aigateway serve --host 127.0.0.1 --port 0 --config gateway.toml
//! ```
//!
//! With `--port 0` the OS assigns a free port; the actual bound address is
//! printed to stdout as `listening on http://127.0.0.1:<port>` so a spawning
//! daemon can read it.

use std::io::Write;
use std::path::PathBuf;

use aigateway::{AppState, GatewayConfig, Upstream, serve};
use anyhow::Context;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "aigateway",
    version,
    about = "Loopback Anthropic → OpenAI-compatible AI gateway"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the loopback gateway server.
    Serve(ServeArgs),
}

#[derive(clap::Args)]
struct ServeArgs {
    /// Address to bind. Defaults to loopback.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Port to bind. `0` lets the OS pick a free port.
    #[arg(long, default_value_t = 0)]
    port: u16,
    /// Path to the TOML config file.
    #[arg(long)]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => run_serve(args).await,
    }
}

async fn run_serve(args: ServeArgs) -> anyhow::Result<()> {
    let config = GatewayConfig::load(&args.config)
        .with_context(|| format!("loading config {}", args.config.display()))?;
    let upstream = Upstream::new(config.upstream)?;
    let state = AppState::new(upstream);

    let bind_addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("binding {bind_addr}"))?;
    let local_addr = listener
        .local_addr()
        .context("resolving bound local address")?;

    // Contract with the spawning daemon: print the real address (with the
    // OS-assigned port when `--port 0`) and flush so it's readable immediately.
    println!("listening on http://{local_addr}");
    std::io::stdout().flush().context("flushing stdout")?;

    serve(listener, state).await
}
