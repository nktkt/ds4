//! Server-Sent Events helpers. The original uses a small inline buffer per
//! stream; here we lean on `hyper`'s streaming `Body`.

use bytes::Bytes;

pub fn sse_chunk(event: &str, data: &str) -> Bytes {
    let mut s = String::with_capacity(event.len() + data.len() + 16);
    if !event.is_empty() {
        s.push_str("event: ");
        s.push_str(event);
        s.push('\n');
    }
    for line in data.split('\n') {
        s.push_str("data: ");
        s.push_str(line);
        s.push('\n');
    }
    s.push('\n');
    Bytes::from(s.into_bytes())
}

pub fn sse_done() -> Bytes {
    Bytes::from_static(b"data: [DONE]\n\n")
}
