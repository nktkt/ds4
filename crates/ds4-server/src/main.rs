//! ds4-server — DwarfStar 4 HTTP server entry point (port of `ds4_server.c`).

fn main() -> anyhow::Result<()> {
    ds4_server::run()
}
