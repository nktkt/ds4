//! Think-mode helpers (`ds4_think_mode_*`). The original C versions live near
//! the top of `ds4.c`; we keep them in their own module so callers don't have
//! to pull in the full engine to interpret a mode string.

use crate::api::ThinkMode;

/// Parse a `--think` style value. Empty/missing maps to `None`.
pub fn parse_mode(s: &str) -> Option<ThinkMode> {
    Some(match s.trim().to_ascii_lowercase().as_str() {
        "" | "off" | "none" | "no" | "false" | "0" => ThinkMode::None,
        "high" | "on" | "true" | "1" => ThinkMode::High,
        "max" | "full" => ThinkMode::Max,
        _ => return None,
    })
}
