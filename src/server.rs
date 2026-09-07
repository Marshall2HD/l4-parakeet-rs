#![cfg_attr(not(all(feature = "cuda", target_os = "linux")), allow(dead_code))]

use serde_json::{Value, json};
use std::error::Error;

pub const DEFAULT_MAX_UPLOAD_BYTES: usize = 256 * 1024 * 1024;
const MODEL_ID: &str = "parakeet-tdt-0.6b-v2";
const MAX_MULTIPART_PARTS: usize = 64;
const MAX_PART_HEADER_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResponseFormat {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

#[derive(Debug)]
struct ApiError {
    status: u16,
    message: String,
    error_type: &'static str,
    param: Option<&'static str>,
    code: &'static str,
}

impl ApiError {
    fn invalid(
        message: impl Into<String>,
        param: Option<&'static str>,
        code: &'static str,
    ) -> Self {
        Self {
            status: 400,
            message: message.into(),
            error_type: "invalid_request_error",
            param,
            code,
        }
    }

    fn payload_too_large(limit: usize) -> Self {
        Self {
            status: 413,
            message: format!("request body exceeds the configured {limit}-byte upload limit"),
            error_type: "invalid_request_error",
            param: Some("file"),
            code: "payload_too_large",
        }
    }

    fn internal() -> Self {
        Self {
            status: 500,
            message: "internal transcription error".into(),
            error_type: "server_error",
            param: None,
            code: "internal_error",
        }
    }

    fn not_found() -> Self {
        Self {
            status: 404,
            message: "route not found".into(),
            error_type: "invalid_request_error",
            param: None,
            code: "not_found",
        }
    }

    fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: 501,
            message: message.into(),
            error_type: "invalid_request_error",
            param: None,
            code: "not_implemented",
        }
    }
}

#[derive(Debug)]
struct ApiResponse {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
}

impl ApiResponse {
    fn json(value: Value) -> Self {
        Self {
            status: 200,
            content_type: "application/json",
            body: serde_json::to_vec(&value).expect("JSON value is serializable"),
        }
    }

    fn error(error: ApiError) -> Self {
        Self {
            status: error.status,
            content_type: "application/json",
            body: serde_json::to_vec(&json!({
                "error": {
                    "message": error.message,
                    "type": error.error_type,
                    "param": error.param,
                    "code": error.code,
                }
            }))
            .expect("JSON value is serializable"),
        }
    }
}

#[derive(Debug)]
struct MultipartPart<'a> {
    name: String,
    data: &'a [u8],
}

#[derive(Debug)]
struct TranscriptionForm<'a> {
    audio: &'a [u8],
    format: ResponseFormat,
    include_words: bool,
}

fn parse_transcription_form<'a>(
    content_type: &str,
    body: &'a [u8],
) -> Result<TranscriptionForm<'a>, ApiError> {
    let boundary = multipart_boundary(content_type)?;
    let parts = parse_multipart(body, &boundary)?;
    let files = parts
        .iter()
        .filter(|part| part.name == "file")
        .collect::<Vec<_>>();
    if files.is_empty() {
        return Err(ApiError::invalid(
            "missing required field 'file'",
            Some("file"),
            "missing_field",
        ));
    }
    if files.len() != 1 {
        return Err(ApiError::invalid(
            "field 'file' must occur exactly once",
            Some("file"),
            "duplicate_field",
        ));
    }

    let format = match optional_text_field(&parts, "response_format")?.as_deref() {
        None | Some("") | Some("json") => ResponseFormat::Json,
        Some("text") => ResponseFormat::Text,
        Some("verbose_json") => ResponseFormat::VerboseJson,
        Some("srt") => ResponseFormat::Srt,
        Some("vtt") => ResponseFormat::Vtt,
        Some(value) => {
            return Err(ApiError::invalid(
                format!("unsupported response_format '{value}'"),
                Some("response_format"),
                "unsupported_response_format",
            ));
        }
    };
    let mut include_words = false;
    for part in parts.iter().filter(|part| {
        part.name == "timestamp_granularities[]" || part.name == "timestamp_granularities"
    }) {
        let value = text_value(part, "timestamp_granularities[]")?;
        match value {
            "word" => include_words = true,
            "segment" => {}
            _ => {
                return Err(ApiError::invalid(
                    format!("unsupported timestamp granularity '{value}'"),
                    Some("timestamp_granularities[]"),
                    "unsupported_timestamp_granularity",
                ));
            }
        }
    }

    Ok(TranscriptionForm {
        audio: files[0].data,
        format,
        include_words,
    })
}

fn multipart_boundary(content_type: &str) -> Result<String, ApiError> {
    let mut fields = content_type.split(';');
    if !fields
        .next()
        .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("multipart/form-data"))
    {
        return Err(ApiError::invalid(
            "content type must be multipart/form-data",
            None,
            "invalid_content_type",
        ));
    }
    let boundary = fields.find_map(|field| {
        let (name, value) = field.trim().split_once('=')?;
        name.trim()
            .eq_ignore_ascii_case("boundary")
            .then(|| value.trim().trim_matches('"').to_owned())
    });
    let Some(boundary) = boundary else {
        return Err(ApiError::invalid(
            "multipart boundary is missing",
            None,
            "invalid_multipart",
        ));
    };
    if boundary.is_empty()
        || boundary.len() > 70
        || boundary.bytes().any(|byte| byte <= b' ' || byte >= 0x7f)
    {
        return Err(ApiError::invalid(
            "multipart boundary is invalid",
            None,
            "invalid_multipart",
        ));
    }
    Ok(boundary)
}

fn parse_multipart<'a>(body: &'a [u8], boundary: &str) -> Result<Vec<MultipartPart<'a>>, ApiError> {
    let delimiter = [b"--".as_slice(), boundary.as_bytes()].concat();
    if !body.starts_with(&delimiter) {
        return Err(ApiError::invalid(
            "multipart body does not start with its boundary",
            None,
            "invalid_multipart",
        ));
    }
    let next_delimiter = [b"\r\n".as_slice(), delimiter.as_slice()].concat();
    let mut parts = Vec::new();
    let mut cursor = delimiter.len();
    loop {
        if body.get(cursor..cursor + 2) == Some(b"--") {
            cursor += 2;
            if body.get(cursor..).is_some_and(|tail| {
                tail.is_empty() || tail == b"\r\n" || tail.iter().all(u8::is_ascii_whitespace)
            }) {
                return Ok(parts);
            }
            return Err(ApiError::invalid(
                "unexpected bytes after final multipart boundary",
                None,
                "invalid_multipart",
            ));
        }
        if body.get(cursor..cursor + 2) != Some(b"\r\n") {
            return Err(ApiError::invalid(
                "multipart boundary is not followed by CRLF",
                None,
                "invalid_multipart",
            ));
        }
        cursor += 2;
        if parts.len() == MAX_MULTIPART_PARTS {
            return Err(ApiError::invalid(
                "multipart body contains too many fields",
                None,
                "invalid_multipart",
            ));
        }
        let header_end = find_bytes(&body[cursor..], b"\r\n\r\n")
            .map(|offset| cursor + offset)
            .ok_or_else(|| {
                ApiError::invalid(
                    "multipart field headers are incomplete",
                    None,
                    "invalid_multipart",
                )
            })?;
        if header_end - cursor > MAX_PART_HEADER_BYTES {
            return Err(ApiError::invalid(
                "multipart field headers are too large",
                None,
                "invalid_multipart",
            ));
        }
        let name = part_name(&body[cursor..header_end])?;
        let data_start = header_end + 4;
        let data_end = find_bytes(&body[data_start..], &next_delimiter)
            .map(|offset| data_start + offset)
            .ok_or_else(|| {
                ApiError::invalid(
                    "multipart field is missing a closing boundary",
                    None,
                    "invalid_multipart",
                )
            })?;
        parts.push(MultipartPart {
            name,
            data: &body[data_start..data_end],
        });
        cursor = data_end + next_delimiter.len();
    }
}

fn part_name(headers: &[u8]) -> Result<String, ApiError> {
    let headers = std::str::from_utf8(headers).map_err(|_| {
        ApiError::invalid(
            "multipart field headers are not UTF-8",
            None,
            "invalid_multipart",
        )
    })?;
    let disposition = headers
        .split("\r\n")
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("content-disposition")
                .then_some(value.trim())
        })
        .ok_or_else(|| {
            ApiError::invalid(
                "multipart field is missing Content-Disposition",
                None,
                "invalid_multipart",
            )
        })?;
    let mut fields = disposition.split(';');
    if !fields
        .next()
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("form-data"))
    {
        return Err(ApiError::invalid(
            "multipart Content-Disposition must be form-data",
            None,
            "invalid_multipart",
        ));
    }
    fields
        .find_map(|field| {
            let (key, value) = field.trim().split_once('=')?;
            key.eq_ignore_ascii_case("name")
                .then(|| value.trim().trim_matches('"').to_owned())
        })
        .filter(|name| !name.is_empty())
        .ok_or_else(|| ApiError::invalid("multipart field has no name", None, "invalid_multipart"))
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    (!needle.is_empty() && haystack.len() >= needle.len())
        .then(|| {
            haystack[..haystack.len() - needle.len() + 1]
                .chunks(64)
                .enumerate()
                .filter(|(_, chunk)| chunk.contains(&needle[0]))
                .find_map(|(chunk_index, chunk)| {
                    chunk.iter().enumerate().find_map(|(index, &byte)| {
                        let offset = chunk_index * 64 + index;
                        (byte == needle[0] && haystack[offset..].starts_with(needle))
                            .then_some(offset)
                    })
                })
        })
        .flatten()
}

fn optional_text_field(
    parts: &[MultipartPart<'_>],
    name: &'static str,
) -> Result<Option<String>, ApiError> {
    let values = parts
        .iter()
        .filter(|part| part.name == name)
        .collect::<Vec<_>>();
    match values.as_slice() {
        [] => Ok(None),
        [part] => Ok(Some(text_value(part, name)?.to_owned())),
        _ => Err(ApiError::invalid(
            format!("field '{name}' must not be repeated"),
            Some(name),
            "duplicate_field",
        )),
    }
}

fn text_value<'a>(part: &'a MultipartPart<'_>, param: &'static str) -> Result<&'a str, ApiError> {
    if part.data.len() > 4_096 {
        return Err(ApiError::invalid(
            format!("field '{param}' is too large"),
            Some(param),
            "invalid_field",
        ));
    }
    std::str::from_utf8(part.data).map_err(|_| {
        ApiError::invalid(
            format!("field '{param}' is not UTF-8"),
            Some(param),
            "invalid_field",
        )
    })
}

fn format_transcription(
    text: &str,
    duration: f64,
    format: ResponseFormat,
    include_words: bool,
) -> ApiResponse {
    match format {
        ResponseFormat::Json => ApiResponse::json(json!({ "text": text })),
        ResponseFormat::Text => ApiResponse {
            status: 200,
            content_type: "text/plain; charset=utf-8",
            body: text.as_bytes().to_vec(),
        },
        ResponseFormat::VerboseJson => {
            let words = text.split_whitespace().collect::<Vec<_>>();
            let mut value = json!({
                "task": "transcribe",
                "language": "en",
                "duration": duration,
                "text": text,
                "segments": [{
                    "id": 0,
                    "start": 0.0,
                    "end": duration,
                    "text": text,
                }],
            });
            if include_words {
                let count = words.len() as f64;
                value["words"] = Value::Array(
                    words
                        .into_iter()
                        .enumerate()
                        .map(|(index, word)| {
                            json!({
                                "word": word,
                                "start": duration * index as f64 / count,
                                "end": duration * (index + 1) as f64 / count,
                            })
                        })
                        .collect(),
                );
            }
            ApiResponse::json(value)
        }
        ResponseFormat::Srt => ApiResponse {
            status: 200,
            content_type: "text/plain; charset=utf-8",
            body: format!(
                "1\n{} --> {}\n{text}\n",
                subtitle_timestamp(0.0, ','),
                subtitle_timestamp(duration, ',')
            )
            .into_bytes(),
        },
        ResponseFormat::Vtt => ApiResponse {
            status: 200,
            content_type: "text/vtt; charset=utf-8",
            body: format!(
                "WEBVTT\n\n{} --> {}\n{text}\n",
                subtitle_timestamp(0.0, '.'),
                subtitle_timestamp(duration, '.')
            )
            .into_bytes(),
        },
    }
}

fn subtitle_timestamp(seconds: f64, separator: char) -> String {
    let total_milliseconds = (seconds.max(0.0) * 1_000.0).round() as u64;
    let milliseconds = total_milliseconds % 1_000;
    let total_seconds = total_milliseconds / 1_000;
    let seconds = total_seconds % 60;
    let total_minutes = total_seconds / 60;
    let minutes = total_minutes % 60;
    let hours = total_minutes / 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}")
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub fn serve(
    artifact: &std::path::Path,
    host: &str,
    port: u16,
    device: usize,
    max_audio_seconds: usize,
    max_upload_bytes: usize,
) -> Result<(), Box<dyn Error>> {
    use crate::audio::decode_pcm16_mono_into;
    use crate::cuda::PipelineEngine;
    use std::io::Read;
    use tiny_http::{Header, Method, Response, Server, StatusCode};

    let max_samples = max_audio_seconds
        .checked_mul(16_000)
        .ok_or("maximum audio sample count overflow")?;
    if max_samples == 0 || max_upload_bytes == 0 {
        return Err("maximum audio duration and upload bytes must be nonzero".into());
    }
    let mut engine = PipelineEngine::load(device, artifact, max_samples)?;
    let address = format!("{host}:{port}");
    let server = Server::http(&address)
        .map_err(|error| format!("failed to bind HTTP server at {address}: {error}"))?;
    eprintln!("parakeet-l4: listening on http://{address} ({MODEL_ID}, max {max_audio_seconds}s)");

    let mut body = Vec::new();
    let mut samples = Vec::new();
    let mut sequence = 0_u64;
    for mut request in server.incoming_requests() {
        sequence = sequence.wrapping_add(1);
        let request_id = format!("parakeet-{}-{sequence:016x}", std::process::id());
        let method = request.method().clone();
        let path = request
            .url()
            .split('?')
            .next()
            .unwrap_or(request.url())
            .to_owned();
        let result = match (method, path.as_str()) {
            (Method::Get, "/health" | "/healthz" | "/v1/health/ready") => Ok(ApiResponse::json(
                json!({ "status": "ok", "model": MODEL_ID }),
            )),
            (Method::Get, "/v1/models" | "/openai/v1/models") => Ok(ApiResponse::json(json!({
                "object": "list",
                "data": [{
                    "id": MODEL_ID,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nvidia",
                }],
            }))),
            (Method::Post, "/v1/audio/translations" | "/openai/v1/audio/translations") => {
                Err(ApiError::not_implemented(
                    "audio translation is not supported by this transcription-only backend",
                ))
            }
            (Method::Post, "/v1/audio/transcriptions" | "/openai/v1/audio/transcriptions") => {
                let content_type = request
                    .headers()
                    .iter()
                    .find(|header| header.field.equiv("Content-Type"))
                    .map(|header| header.value.as_str().to_owned())
                    .unwrap_or_default();
                if request
                    .body_length()
                    .is_some_and(|length| length > max_upload_bytes)
                {
                    Err(ApiError::payload_too_large(max_upload_bytes))
                } else {
                    body.clear();
                    body.reserve(request.body_length().unwrap_or(0).min(max_upload_bytes));
                    let read_result = request
                        .as_reader()
                        .take(u64::try_from(max_upload_bytes)? + 1)
                        .read_to_end(&mut body);
                    match read_result {
                        Err(error) => Err(ApiError::invalid(
                            format!("failed to read request body: {error}"),
                            None,
                            "invalid_request_body",
                        )),
                        Ok(_) if body.len() > max_upload_bytes => {
                            Err(ApiError::payload_too_large(max_upload_bytes))
                        }
                        Ok(_) => match parse_transcription_form(&content_type, &body) {
                            Err(error) => Err(error),
                            Ok(form) => match decode_pcm16_mono_into(form.audio, &mut samples) {
                                Err(error) => Err(ApiError::invalid(
                                    format!(
                                        "could not decode audio; only 16 kHz mono PCM16 WAV is supported: {error}"
                                    ),
                                    Some("file"),
                                    "invalid_audio",
                                )),
                                Ok(sample_rate) if sample_rate != 16_000 => Err(ApiError::invalid(
                                    format!("input sample rate is {sample_rate}, expected 16000"),
                                    Some("file"),
                                    "unsupported_sample_rate",
                                )),
                                Ok(_) if samples.len() > engine.max_samples() => {
                                    Err(ApiError::invalid(
                                        format!(
                                            "audio duration exceeds the configured {max_audio_seconds}-second maximum"
                                        ),
                                        Some("file"),
                                        "audio_too_long",
                                    ))
                                }
                                Ok(_) => match engine.transcribe(&samples) {
                                    Ok(transcription) => {
                                        eprintln!(
                                            "parakeet-l4: {request_id} transcribed {:.3}s in {:.3} ms",
                                            transcription.audio_seconds,
                                            transcription.inference_latency_ms
                                        );
                                        Ok(format_transcription(
                                            &transcription.text,
                                            transcription.audio_seconds,
                                            form.format,
                                            form.include_words,
                                        ))
                                    }
                                    Err(error) => {
                                        eprintln!(
                                            "parakeet-l4: {request_id} inference failed: {error}"
                                        );
                                        Err(ApiError::internal())
                                    }
                                },
                            },
                        },
                    }
                }
            }
            _ => Err(ApiError::not_found()),
        };
        let response = result.unwrap_or_else(ApiResponse::error);
        let content_type = Header::from_bytes("Content-Type", response.content_type)
            .expect("static content type header is valid");
        let request_id_header = Header::from_bytes("x-request-id", request_id)
            .expect("generated request ID header is valid");
        let response = Response::from_data(response.body)
            .with_status_code(StatusCode(response.status))
            .with_header(content_type)
            .with_header(request_id_header);
        if let Err(error) = request.respond(response) {
            eprintln!("parakeet-l4: failed to write HTTP response: {error}");
        }
    }
    Ok(())
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
#[allow(clippy::too_many_arguments)]
pub fn serve(
    _artifact: &std::path::Path,
    _host: &str,
    _port: u16,
    _device: usize,
    _max_audio_seconds: usize,
    _max_upload_bytes: usize,
) -> Result<(), Box<dyn Error>> {
    Err("CUDA support requires x86_64 Linux and `cargo build --release --features cuda`".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multipart(fields: &[(&str, &[u8])]) -> (String, Vec<u8>) {
        let boundary = "parakeet-test-boundary";
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            body.extend_from_slice(value);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        (format!("multipart/form-data; boundary={boundary}"), body)
    }

    #[test]
    fn parses_binary_file_and_openai_fields() {
        let audio = b"RIFF\0\xff\r\n--not-the-real-boundary";
        let (content_type, body) = multipart(&[
            ("model", b"whisper-1"),
            ("file", audio),
            ("response_format", b"verbose_json"),
            ("timestamp_granularities[]", b"word"),
        ]);
        let form = parse_transcription_form(&content_type, &body).unwrap();
        assert_eq!(form.audio, audio);
        assert_eq!(form.format, ResponseFormat::VerboseJson);
        assert!(form.include_words);
    }

    #[test]
    fn finds_binary_delimiters_across_scan_chunks() {
        let needle = b"\r\n--parakeet-boundary";
        for offset in 0..192 {
            let mut body = vec![b'\r'; offset];
            body.extend_from_slice(needle);
            body.extend_from_slice(needle);
            assert_eq!(find_bytes(&body, needle), Some(offset));
            body.truncate(offset + needle.len() - 1);
            assert_eq!(find_bytes(&body, needle), None);
        }
        assert_eq!(find_bytes(b"", needle), None);
        assert_eq!(find_bytes(needle, b""), None);
        assert_eq!(find_bytes(b"abc\0\xff", b"\0\xff"), Some(3));
        assert_eq!(find_bytes(b"abc", b"c"), Some(2));
    }

    #[test]
    fn rejects_missing_file_and_unknown_format() {
        let (content_type, body) = multipart(&[("model", b"whisper-1")]);
        let error = parse_transcription_form(&content_type, &body).unwrap_err();
        assert_eq!(error.status, 400);
        assert_eq!(error.code, "missing_field");

        let (content_type, body) = multipart(&[("file", b"WAV"), ("response_format", b"not-real")]);
        let error = parse_transcription_form(&content_type, &body).unwrap_err();
        assert_eq!(error.param, Some("response_format"));
        assert_eq!(error.code, "unsupported_response_format");
    }

    #[test]
    fn formats_all_whisper_response_shapes() {
        let json = format_transcription("hello world", 2.5, ResponseFormat::Json, false);
        assert_eq!(
            serde_json::from_slice::<Value>(&json.body).unwrap()["text"],
            "hello world"
        );

        let verbose = format_transcription("hello world", 2.5, ResponseFormat::VerboseJson, true);
        let verbose = serde_json::from_slice::<Value>(&verbose.body).unwrap();
        assert_eq!(verbose["language"], "en");
        assert_eq!(verbose["segments"][0]["end"], 2.5);
        assert_eq!(verbose["words"].as_array().unwrap().len(), 2);

        let text = format_transcription("hello", 2.5, ResponseFormat::Text, false);
        assert_eq!(text.body, b"hello");
        let srt = format_transcription("hello", 2.5, ResponseFormat::Srt, false);
        assert!(
            String::from_utf8(srt.body)
                .unwrap()
                .contains("00:00:02,500")
        );
        let vtt = format_transcription("hello", 2.5, ResponseFormat::Vtt, false);
        assert!(String::from_utf8(vtt.body).unwrap().starts_with("WEBVTT"));
    }

    #[test]
    fn emits_full_openai_error_envelope() {
        let response = ApiResponse::error(ApiError::invalid(
            "bad format",
            Some("response_format"),
            "unsupported_response_format",
        ));
        let value = serde_json::from_slice::<Value>(&response.body).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["param"], "response_format");
        assert_eq!(value["error"]["code"], "unsupported_response_format");
    }
}
