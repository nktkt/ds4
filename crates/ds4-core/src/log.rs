//! Logging helpers — Rust mirror of `ds4_log` / `ds4_log_is_tty`.
//!
//! The original CLI and server colorize their log lines based on whether the
//! target stream is a TTY. We keep that affordance: pass `&mut Stdout` /
//! `&mut Stderr` (or any writer that implements [`Tty`]) and the colors come
//! out conditional on `Tty::is_tty()`. The actual log message format matches
//! the C output character-for-character so existing parsers keep working.

use crate::api::LogType;
use std::io::Write;

pub trait Tty: Write {
    fn is_tty(&self) -> bool;
}

#[cfg(unix)]
fn fd_is_tty(fd: libc::c_int) -> bool {
    // SAFETY: isatty is safe to call on any int; result is 0 or 1.
    unsafe { libc::isatty(fd) != 0 }
}

#[cfg(unix)]
impl Tty for std::io::Stdout {
    fn is_tty(&self) -> bool { fd_is_tty(libc::STDOUT_FILENO) }
}
#[cfg(unix)]
impl Tty for std::io::Stderr {
    fn is_tty(&self) -> bool { fd_is_tty(libc::STDERR_FILENO) }
}

/// Returns true if `fp` is a tty. The original takes a `FILE*`; we accept
/// anything that knows how to answer the question.
pub fn is_tty(fp: &dyn Tty) -> bool { fp.is_tty() }

pub fn type_prefix(t: LogType, color: bool) -> &'static str {
    if !color {
        return match t {
            LogType::Default    => "",
            LogType::Prefill    => "[prefill] ",
            LogType::Generation => "[gen] ",
            LogType::KvCache    => "[kv] ",
            LogType::Tool       => "[tool] ",
            LogType::Warning    => "[warn] ",
            LogType::Timing     => "[time] ",
            LogType::Ok         => "[ok] ",
            LogType::Error      => "[err] ",
        };
    }
    match t {
        LogType::Default    => "",
        LogType::Prefill    => "\x1b[36m[prefill]\x1b[0m ",
        LogType::Generation => "\x1b[35m[gen]\x1b[0m ",
        LogType::KvCache    => "\x1b[34m[kv]\x1b[0m ",
        LogType::Tool       => "\x1b[33m[tool]\x1b[0m ",
        LogType::Warning    => "\x1b[33m[warn]\x1b[0m ",
        LogType::Timing     => "\x1b[36m[time]\x1b[0m ",
        LogType::Ok         => "\x1b[32m[ok]\x1b[0m ",
        LogType::Error      => "\x1b[31m[err]\x1b[0m ",
    }
}

pub fn write_log<W: Tty + ?Sized>(fp: &mut W, t: LogType, msg: &str) -> std::io::Result<()> {
    let prefix = type_prefix(t, fp.is_tty());
    write!(fp, "{prefix}{msg}")
}

/// Convenience for `ds4_log(fp, type, "format", args...)` — write a line and
/// append a newline if the message doesn't already end with one.
pub fn line<W: Tty + ?Sized>(fp: &mut W, t: LogType, msg: &str) -> std::io::Result<()> {
    let needs_nl = !msg.ends_with('\n');
    write_log(fp, t, msg)?;
    if needs_nl { fp.write_all(b"\n")?; }
    Ok(())
}
