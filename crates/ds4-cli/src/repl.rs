//! Interactive REPL. Mirrors the loop in `ds4_cli.c::repl()` but uses
//! `rustyline` for line editing instead of the bundled `linenoise.c`.

use crate::args::Cli;
use anyhow::Result;
use ds4_core::{Engine, ThinkMode, Tokens};
use rustyline::DefaultEditor;

pub fn run(engine: &Engine, cli: &Cli, think: ThinkMode) -> Result<()> {
    let mut rl = DefaultEditor::new()?;
    let _ = rl.load_history(".ds4_history");
    let mut transcript = crate::transcript::Transcript::new();
    let mut session = engine.new_session(cli.ctx_size.max(0) as u32)?;
    loop {
        let line = match rl.readline("> ") {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Eof)
            | Err(rustyline::error::ReadlineError::Interrupted) => break,
            Err(e) => return Err(e.into()),
        };
        if line.trim().is_empty() { continue; }
        let _ = rl.add_history_entry(&line);
        transcript.push_user(&line);
        let mut prompt = Tokens::new();
        transcript.render(engine, think, &mut prompt);
        session.sync(&prompt)?;
        let mut out_bytes: Vec<u8> = Vec::new();
        for _ in 0..cli.n_predict {
            let t = session.argmax();
            if t == engine.tokenizer().eos_id() { break; }
            if let Some(bytes) = engine.tokenizer().token_text(t) {
                out_bytes.extend_from_slice(bytes);
            }
            session.eval(t)?;
        }
        let answer = String::from_utf8_lossy(&out_bytes).into_owned();
        println!("{answer}");
        transcript.push_assistant(&answer);
    }
    let _ = rl.save_history(".ds4_history");
    Ok(())
}

pub fn run_oneshot(engine: &Engine, prompt: &str, cli: &Cli, think: ThinkMode) -> Result<()> {
    let mut transcript = crate::transcript::Transcript::new();
    transcript.push_user(prompt);
    let mut tokens = Tokens::new();
    transcript.render(engine, think, &mut tokens);
    let mut emit_bytes: Vec<u8> = Vec::new();
    let eos = engine.tokenizer().eos_id();
    let mut sink = |t: i32| {
        if t == eos { return; }
        if let Some(b) = engine.tokenizer().token_text(t) {
            emit_bytes.extend_from_slice(b);
        }
    };
    engine.generate_argmax(&tokens, cli.n_predict, cli.ctx_size, &mut sink, None)?;
    println!("{}", String::from_utf8_lossy(&emit_bytes));
    Ok(())
}
