use super::*;
use crate::message::{ContentBlock, Message, Role};
use crate::provider::{EventStream, Provider};
use crate::tool::Registry;
use async_trait::async_trait;
use futures::StreamExt;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;

struct MockProvider;

#[async_trait]
impl Provider for MockProvider {
    async fn complete(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _system: &str,
        _resume_session_id: Option<&str>,
    ) -> anyhow::Result<EventStream> {
        Err(anyhow::anyhow!(
            "Mock provider should not be used for streaming completions in Gemini tests"
        ))
    }

    fn name(&self) -> &str {
        "mock"
    }

    fn fork(&self) -> Arc<dyn Provider> {
        Arc::new(MockProvider)
    }
}

#[test]
fn available_models_include_gemini_defaults() {
    let provider = GeminiProvider::new();
    let models = provider.available_models();
    assert!(models.contains(&"gemini-3-pro-preview"));
    assert!(models.contains(&"gemini-3.1-pro-preview"));
    assert!(models.contains(&"gemini-2.5-pro"));
    assert!(models.contains(&"gemini-2.5-flash"));
}

#[test]
fn set_model_accepts_gemini_models() {
    let provider = GeminiProvider::new();
    provider.set_model("gemini-2.5-flash").unwrap();
    assert_eq!(provider.model(), "gemini-2.5-flash");
}

#[test]
fn detects_model_not_found_errors() {
    let err = anyhow::anyhow!(
        "Gemini request generateContent failed (HTTP 404 Not Found): {{\"error\":{{\"status\":\"NOT_FOUND\",\"message\":\"Requested entity was not found.\"}}}}"
    );
    assert!(is_gemini_model_not_found_error(&err));
}

#[test]
fn fallback_models_skip_current_model() {
    assert_eq!(
        gemini_fallback_models("gemini-2.5-flash"),
        vec![
            "gemini-3.1-pro-preview",
            "gemini-3-pro-preview",
            "gemini-2.5-pro",
            "gemini-3-flash-preview",
            "gemini-2.0-flash",
        ]
    );
}

#[test]
fn extract_gemini_model_ids_discovers_nested_models() {
    let response = json!({
        "routing": {
            "manual": {
                "models": [
                    {"id": "gemini-3-pro-preview"},
                    {"name": "gemini-3.1-pro-preview"}
                ]
            },
            "auto": ["gemini-3-flash-preview", "not-a-model"]
        }
    });

    assert_eq!(
        extract_gemini_model_ids(&response),
        vec![
            "gemini-3.1-pro-preview".to_string(),
            "gemini-3-pro-preview".to_string(),
            "gemini-3-flash-preview".to_string(),
        ]
    );
}

#[test]
fn available_models_display_prefers_discovered_models_and_current_model() {
    let provider = GeminiProvider::new();
    provider.set_model("gemini-4-pro-preview").unwrap();
    *provider.fetched_models.write().unwrap() = vec![
        "gemini-3-flash-preview".to_string(),
        "gemini-3-pro-preview".to_string(),
    ];

    assert_eq!(
        provider.available_models_display(),
        vec![
            "gemini-3-pro-preview".to_string(),
            "gemini-3-flash-preview".to_string(),
            "gemini-4-pro-preview".to_string(),
        ]
    );
}

#[test]
fn available_models_display_without_discovery_uses_current_model_only() {
    let provider = GeminiProvider::new();
    provider.set_model("gemini-4-pro-preview").unwrap();

    assert_eq!(
        provider.available_models_display(),
        vec!["gemini-4-pro-preview".to_string()]
    );
}

#[test]
fn available_models_display_seeds_from_persisted_catalog() {
    let _guard = crate::storage::lock_test_env();
    let temp = tempfile::TempDir::new().expect("tempdir");
    let prev_home = std::env::var_os("JCODE_HOME");
    crate::env::set_var("JCODE_HOME", temp.path());

    let path = GeminiProvider::persisted_catalog_path().expect("catalog path");
    crate::storage::write_json(
        &path,
        &PersistedCatalog {
            models: vec!["gemini-3-pro-preview".to_string()],
            fetched_at_rfc3339: chrono::Utc::now().to_rfc3339(),
        },
    )
    .expect("write persisted catalog");

    let provider = GeminiProvider::new();
    assert!(
        provider
            .available_models_display()
            .contains(&"gemini-3-pro-preview".to_string())
    );

    if let Some(prev_home) = prev_home {
        crate::env::set_var("JCODE_HOME", prev_home);
    } else {
        crate::env::remove_var("JCODE_HOME");
    }
}

#[tokio::test]
async fn prefetch_models_with_api_key_uses_public_model_list_without_oauth() {
    let _guard = crate::storage::lock_test_env();
    let temp = tempfile::TempDir::new().expect("tempdir");
    let prev_home = std::env::var_os("JCODE_HOME");
    let prev_key = std::env::var_os(crate::auth::gemini::GEMINI_API_KEY_ENV);
    let prev_endpoint = std::env::var_os("GEMINI_API_ENDPOINT");
    let prev_version = std::env::var_os("GEMINI_API_VERSION");

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let addr = listener.local_addr().expect("server addr");
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept model list request");
        let mut request = [0; 2048];
        let n = stream.read(&mut request).expect("read request");
        let request = String::from_utf8_lossy(&request[..n]);
        assert!(request.starts_with("GET /v1beta/models?key=gemini-test-key HTTP/1.1"));
        let body = serde_json::json!({
            "models": [
                {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]}
            ]
        })
        .to_string();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        )
        .expect("write response");
    });

    crate::env::set_var("JCODE_HOME", temp.path());
    crate::env::set_var(crate::auth::gemini::GEMINI_API_KEY_ENV, "gemini-test-key");
    crate::env::set_var("GEMINI_API_ENDPOINT", format!("http://{addr}"));
    crate::env::set_var("GEMINI_API_VERSION", "v1beta");

    let provider = GeminiProvider::new();
    provider.prefetch_models().await.expect("prefetch models");
    assert!(
        provider
            .available_models_display()
            .contains(&"gemini-2.5-flash".to_string())
    );
    handle.join().expect("server thread");

    if let Some(prev_home) = prev_home {
        crate::env::set_var("JCODE_HOME", prev_home);
    } else {
        crate::env::remove_var("JCODE_HOME");
    }
    if let Some(prev_key) = prev_key {
        crate::env::set_var(crate::auth::gemini::GEMINI_API_KEY_ENV, prev_key);
    } else {
        crate::env::remove_var(crate::auth::gemini::GEMINI_API_KEY_ENV);
    }
    if let Some(prev_endpoint) = prev_endpoint {
        crate::env::set_var("GEMINI_API_ENDPOINT", prev_endpoint);
    } else {
        crate::env::remove_var("GEMINI_API_ENDPOINT");
    }
    if let Some(prev_version) = prev_version {
        crate::env::set_var("GEMINI_API_VERSION", prev_version);
    } else {
        crate::env::remove_var("GEMINI_API_VERSION");
    }
}

#[tokio::test]
async fn complete_with_api_key_uses_public_generate_content_without_oauth_setup() {
    let _guard = crate::storage::lock_test_env();
    let temp = tempfile::TempDir::new().expect("tempdir");
    let prev_home = std::env::var_os("JCODE_HOME");
    let prev_key = std::env::var_os(crate::auth::gemini::GEMINI_API_KEY_ENV);
    let prev_endpoint = std::env::var_os("GEMINI_API_ENDPOINT");
    let prev_version = std::env::var_os("GEMINI_API_VERSION");

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let addr = listener.local_addr().expect("server addr");
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept generate request");
        let mut request = [0; 4096];
        let n = stream.read(&mut request).expect("read request");
        let request = String::from_utf8_lossy(&request[..n]);
        assert!(request.starts_with(
            "POST /v1beta/models/gemini-2.5-flash:generateContent?key=gemini-test-key HTTP/1.1"
        ));
        assert!(request.contains("hello gemini"));
        let body = serde_json::json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "hello from api key"}]},
                "finishReason": "STOP"
            }]
        })
        .to_string();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        )
        .expect("write response");
    });

    crate::env::set_var("JCODE_HOME", temp.path());
    crate::env::set_var(crate::auth::gemini::GEMINI_API_KEY_ENV, "gemini-test-key");
    crate::env::set_var("GEMINI_API_ENDPOINT", format!("http://{addr}"));
    crate::env::set_var("GEMINI_API_VERSION", "v1beta");

    let provider = GeminiProvider::new();
    provider.set_model("gemini-2.5-flash").expect("set model");
    let messages = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::Text {
            text: "hello gemini".to_string(),
            cache_control: None,
        }],
        timestamp: None,
        tool_duration_ms: None,
    }];
    let mut stream = provider
        .complete(&messages, &[], "", None)
        .await
        .expect("complete stream");
    let mut saw_text = false;
    while let Some(event) = stream.next().await {
        if let StreamEvent::TextDelta(text) = event.expect("stream event") {
            assert_eq!(text, "hello from api key");
            saw_text = true;
        }
    }
    assert!(saw_text);
    handle.join().expect("server thread");

    if let Some(prev_home) = prev_home {
        crate::env::set_var("JCODE_HOME", prev_home);
    } else {
        crate::env::remove_var("JCODE_HOME");
    }
    if let Some(prev_key) = prev_key {
        crate::env::set_var(crate::auth::gemini::GEMINI_API_KEY_ENV, prev_key);
    } else {
        crate::env::remove_var(crate::auth::gemini::GEMINI_API_KEY_ENV);
    }
    if let Some(prev_endpoint) = prev_endpoint {
        crate::env::set_var("GEMINI_API_ENDPOINT", prev_endpoint);
    } else {
        crate::env::remove_var("GEMINI_API_ENDPOINT");
    }
    if let Some(prev_version) = prev_version {
        crate::env::set_var("GEMINI_API_VERSION", prev_version);
    } else {
        crate::env::remove_var("GEMINI_API_VERSION");
    }
}

#[test]
fn build_contents_preserves_tool_calls_and_results() {
    let messages = vec![
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read".to_string(),
                input: json!({"path":"README.md"}),
            }],
            timestamp: None,
            tool_duration_ms: None,
        },
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "ok".to_string(),
                is_error: None,
            }],
            timestamp: None,
            tool_duration_ms: None,
        },
    ];

    let contents = build_contents(&messages);
    assert_eq!(contents.len(), 2);
    assert_eq!(contents[0].role, "model");
    assert_eq!(contents[1].role, "user");
    assert_eq!(
        contents[0].parts[0].function_call.as_ref().unwrap().name,
        "read"
    );
    assert_eq!(
        contents[1].parts[0]
            .function_response
            .as_ref()
            .unwrap()
            .name,
        "read"
    );
}

#[test]
fn build_tools_uses_function_declarations() {
    let defs = vec![ToolDefinition {
        name: "read".to_string(),
        description: "Read a file".to_string(),
        input_schema: json!({"type":"object","properties":{"path":{"type":"string"}}}),
    }];

    let built = build_tools(&defs).unwrap();
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].function_declarations[0].name, "read");
}

fn schema_contains_key(schema: &Value, key: &str) -> bool {
    match schema {
        Value::Object(map) => {
            map.contains_key(key) || map.values().any(|value| schema_contains_key(value, key))
        }
        Value::Array(items) => items.iter().any(|value| schema_contains_key(value, key)),
        _ => false,
    }
}

#[test]
fn build_tools_rewrites_const_for_gemini_schema_compatibility() {
    let defs = vec![ToolDefinition {
        name: "batch".to_string(),
        description: "Batch tools".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "tool": { "type": "string", "const": "read" },
                                    "file_path": { "type": "string" }
                                },
                                "required": ["tool", "file_path"]
                            }
                        ]
                    }
                }
            }
        }),
    }];

    let built = build_tools(&defs).expect("gemini tools");
    let parameters = &built[0].function_declarations[0].parameters;

    assert!(!schema_contains_key(parameters, "const"));
    assert_eq!(
        parameters["properties"]["tool_calls"]["items"]["oneOf"][0]["properties"]["tool"]["enum"],
        json!(["read"])
    );
}

#[tokio::test]
async fn build_tools_from_registry_definitions_omits_const_keywords() {
    let provider: Arc<dyn Provider> = Arc::new(MockProvider);
    let registry = Registry::new(provider).await;
    let defs = registry.definitions(None).await;

    let built = build_tools(&defs).expect("gemini tools");
    let parameters = &built[0].function_declarations;

    assert!(!schema_contains_key(&json!(parameters), "const"));
}

#[test]
fn parses_prompt_feedback_block_reason() {
    let response: VertexGenerateContentResponse = serde_json::from_value(json!({
        "promptFeedback": {
            "blockReason": "PROHIBITED_CONTENT",
            "blockReasonMessage": "Prompt violated policy"
        }
    }))
    .expect("parse prompt feedback");

    let feedback = response.prompt_feedback.expect("missing prompt feedback");
    assert_eq!(feedback.block_reason.as_deref(), Some("PROHIBITED_CONTENT"));
    assert_eq!(
        feedback.block_reason_message.as_deref(),
        Some("Prompt violated policy")
    );
}

#[test]
fn parses_candidate_finish_message() {
    let response: VertexGenerateContentResponse = serde_json::from_value(json!({
        "candidates": [
            {
                "finishReason": "SAFETY",
                "finishMessage": "Response blocked by safety filters"
            }
        ]
    }))
    .expect("parse candidate");

    let candidate = response
        .candidates
        .expect("missing candidates")
        .into_iter()
        .next()
        .expect("missing first candidate");
    assert_eq!(candidate.finish_reason.as_deref(), Some("SAFETY"));
    assert_eq!(
        candidate.finish_message.as_deref(),
        Some("Response blocked by safety filters")
    );
}
