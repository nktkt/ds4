//! HTTP layer. Ported from `ds4_server.c::http_*`.
//!
//! The original is a hand-rolled epoll/kqueue loop with one I/O thread plus
//! N worker threads pulling from a request queue. The Rust port uses tokio +
//! hyper: requests come in as async handlers, then hand off to a sync worker
//! pool that owns the GPU sessions (since the engine itself is not async).

use crate::args::ServerArgs;
use anyhow::Result;
use bytes::Bytes;
use http_body_util::{combinators::BoxBody, BodyExt, Full};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::tokio::TokioIo;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::net::TcpListener;

pub type BoxBod = BoxBody<Bytes, Infallible>;

pub struct AppState {
    pub args: ServerArgs,
    pub queue: crate::queue::JobQueue,
}

pub async fn serve(args: ServerArgs) -> Result<()> {
    let listener = TcpListener::bind(&args.bind).await?;
    log::info!("ds4-server listening on http://{}", args.bind);
    let queue = crate::queue::JobQueue::start(&args)?;
    let state = Arc::new(AppState { args, queue });
    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let st = state.clone();
        tokio::spawn(async move {
            let svc = service_fn(move |req| {
                let st = st.clone();
                async move { route(req, st).await }
            });
            if let Err(e) = http1::Builder::new().serve_connection(io, svc).await {
                log::warn!("connection error: {e}");
            }
        });
    }
}

async fn route(req: Request<Incoming>, st: Arc<AppState>) -> Result<Response<BoxBod>, Infallible> {
    let path = req.uri().path().to_owned();
    let method = req.method().clone();
    let resp = match (method, path.as_str()) {
        (Method::GET, "/v1/models") => Ok(crate::openai::list_models(&st)),
        (Method::POST, "/v1/chat/completions") => crate::openai::chat_completions(req, &st).await,
        (Method::POST, "/v1/completions") => crate::openai::completions(req, &st).await,
        (Method::POST, "/v1/messages") => crate::anthropic::messages(req, &st).await,
        (Method::GET, "/healthz") => Ok(text(200, "ok\n")),
        _ => Ok(text(404, "not found\n")),
    };
    let out = resp.unwrap_or_else(|e| text(500, &format!("error: {e}\n")));
    Ok(out)
}

pub fn text(status: u16, body: &str) -> Response<BoxBod> {
    let body = Full::new(Bytes::from(body.to_owned())).boxed();
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap())
        .header("content-type", "text/plain; charset=utf-8")
        .body(body)
        .unwrap()
}

pub fn json(status: u16, body: serde_json::Value) -> Response<BoxBod> {
    let payload = serde_json::to_vec(&body).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap())
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(payload)).boxed())
        .unwrap()
}
