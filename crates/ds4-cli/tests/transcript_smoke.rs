//! Smoke tests for the CLI transcript builder.
//!
//! These exercise the public `Transcript` API in `ds4_cli::transcript` without
//! requiring a real engine / GGUF file. The render path is covered by other
//! unit tests in `ds4_core::chat`; here we just verify that the higher-level
//! transcript bookkeeping (turn list + optional system prompt) behaves.

use ds4_cli::transcript::{Transcript, Turn};

#[test]
fn pushes_record_turns() {
    let mut t = Transcript::new();
    t.push_user("hello");
    t.push_assistant("hi");

    assert_eq!(t.turns.len(), 2, "two pushes should yield two turns");

    let Turn { role: r0, content: c0 } = &t.turns[0];
    assert_eq!(r0, "user");
    assert_eq!(c0, "hello");

    let Turn { role: r1, content: c1 } = &t.turns[1];
    assert_eq!(r1, "assistant");
    assert_eq!(c1, "hi");
}

#[test]
fn system_set_records() {
    let mut t = Transcript::new();
    assert!(t.system.is_none(), "fresh transcript has no system prompt");

    t.set_system("foo");
    assert_eq!(t.system.as_deref(), Some("foo"));

    // Calling again overwrites the previously stored prompt.
    t.set_system("bar");
    assert_eq!(t.system.as_deref(), Some("bar"));

    // Setting the system prompt must not synthesize any turns.
    assert!(t.turns.is_empty());
}

#[test]
fn transcript_default_empty() {
    let t = Transcript::new();
    assert!(t.turns.is_empty(), "new transcript has no turns");
    assert!(t.system.is_none(), "new transcript has no system prompt");

    // `Default` should match `new()`.
    let d = Transcript::default();
    assert!(d.turns.is_empty());
    assert!(d.system.is_none());
}
