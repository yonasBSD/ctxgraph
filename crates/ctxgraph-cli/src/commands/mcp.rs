use std::env;
use std::path::PathBuf;

use ctxgraph::Graph;
use ctxgraph_embed::EmbedEngine;
use ctxgraph_mcp::McpServer;

/// Resolve the database path from --db flag, env var, or default.
fn resolve_db_path(db: Option<String>) -> PathBuf {
    if let Some(p) = db {
        return PathBuf::from(p);
    }
    if let Ok(val) = env::var("CTXGRAPH_DB") {
        return PathBuf::from(val);
    }
    PathBuf::from(".ctxgraph/graph.db")
}

/// Locate models directory.
fn find_models_dir(db_path: &std::path::Path) -> Option<PathBuf> {
    if let Ok(val) = env::var("CTXGRAPH_MODELS_DIR") {
        let p = PathBuf::from(val);
        if p.is_dir() {
            return Some(p);
        }
    }

    if let Ok(home) = env::var("HOME") {
        let p = PathBuf::from(home).join(".cache/ctxgraph/models");
        if p.is_dir() {
            return Some(p);
        }
    }

    if let Some(ctxgraph_dir) = db_path.parent() {
        let p = ctxgraph_dir.join("models");
        if p.is_dir() {
            return Some(p);
        }
    }

    None
}

pub fn start(db: Option<String>) -> ctxgraph::Result<()> {
    let rt = tokio::runtime::Runtime::new().map_err(ctxgraph::CtxGraphError::Io)?;
    rt.block_on(async {
        let db_path = resolve_db_path(db);
        eprintln!("ctxgraph mcp: using database at {}", db_path.display());

        let mut graph = match Graph::open_or_create(&db_path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("ctxgraph mcp: failed to open/create graph: {e}");
                std::process::exit(1);
            }
        };

        // Load extraction pipeline if models are available
        if let Some(models_dir) = find_models_dir(&db_path) {
            match graph.load_extraction_pipeline(&models_dir) {
                Ok(()) => eprintln!("ctxgraph mcp: extraction pipeline ready"),
                Err(e) => eprintln!("ctxgraph mcp: extraction pipeline not loaded: {e}"),
            }
        }

        // Load embed engine
        let embed = if env::var("CTXGRAPH_NO_EMBED").as_deref() == Ok("1") {
            eprintln!("ctxgraph mcp: embedding disabled (CTXGRAPH_NO_EMBED=1)");
            None
        } else {
            eprintln!("ctxgraph mcp: loading embedding model...");
            match EmbedEngine::new() {
                Ok(e) => {
                    eprintln!("ctxgraph mcp: embedding model ready");
                    Some(e)
                }
                Err(err) => {
                    eprintln!("ctxgraph mcp: warning: embedding unavailable: {err}");
                    None
                }
            }
        };

        eprintln!("ctxgraph mcp: server starting on stdio");
        let server = McpServer::new(graph, embed);
        server.run().await;
    });

    Ok(())
}
