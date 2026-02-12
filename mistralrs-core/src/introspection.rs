//! Public API for activation-level introspection experiments.
//!
//! Provides [`IntrospectionModel`] - a high-level wrapper around model internals
//! that handles weight loading, tokenization, and introspection operations
//! (logit lens, steering vector injection, hidden state capture).
//!
//! Supports multiple model architectures (Qwen3-Next, Qwen2) via auto-detection.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};
use indicatif::MultiProgress;
use mistralrs_quant::ShardedSafeTensors;
use serde::Serialize;
use tokenizers::Tokenizer;

use crate::device_map::DummyDeviceMapper;
use crate::models::{qwen2, qwen3_next};
use crate::models::qwen3_next::{IntrospectionState, MoeRoutingData};
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::{
    text_models_inputs_processor::FlashParams, NormalLoadingMetadata,
};

/// Internal enum dispatching between supported model architectures.
enum ModelBackend {
    Qwen3Next {
        model: qwen3_next::Model,
        config: qwen3_next::Config,
    },
    Qwen2 {
        model: qwen2::Model,
        config: qwen2::Config,
    },
}

impl ModelBackend {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Qwen3Next { model, .. } => {
                model.forward(input_ids, seqlen_offsets, context_lens, metadata, flash_params)
            }
            Self::Qwen2 { model, .. } => {
                model.forward(input_ids, seqlen_offsets, context_lens, metadata, flash_params)
            }
        }
    }

    fn forward_introspect(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<(Tensor, Vec<Tensor>)> {
        match self {
            Self::Qwen3Next { model, .. } => {
                model.forward_introspect(input_ids, seqlen_offsets, context_lens, metadata, flash_params)
            }
            Self::Qwen2 { model, .. } => {
                model.forward_introspect(input_ids, seqlen_offsets, context_lens, metadata, flash_params)
            }
        }
    }

    fn logit_lens_all(&self, hidden_states: &[Tensor]) -> candle_core::Result<Vec<Tensor>> {
        match self {
            Self::Qwen3Next { model, .. } => model.logit_lens_all(hidden_states),
            Self::Qwen2 { model, .. } => model.logit_lens_all(hidden_states),
        }
    }

    fn set_steering_vector(&self, layer_idx: usize, vector: Tensor) {
        match self {
            Self::Qwen3Next { model, .. } => model.set_steering_vector(layer_idx, vector),
            Self::Qwen2 { model, .. } => model.set_steering_vector(layer_idx, vector),
        }
    }

    fn set_steering_vectors_range(
        &self,
        layer_range: std::ops::Range<usize>,
        vector: &Tensor,
        scale: f64,
    ) -> candle_core::Result<()> {
        match self {
            Self::Qwen3Next { model, .. } => model.set_steering_vectors_range(layer_range, vector, scale),
            Self::Qwen2 { model, .. } => model.set_steering_vectors_range(layer_range, vector, scale),
        }
    }

    fn clear_steering_vectors(&self) {
        match self {
            Self::Qwen3Next { model, .. } => model.clear_steering_vectors(),
            Self::Qwen2 { model, .. } => model.clear_steering_vectors(),
        }
    }

    fn introspection(&self) -> &Arc<Mutex<IntrospectionState>> {
        match self {
            Self::Qwen3Next { model, .. } => &model.introspection,
            Self::Qwen2 { model, .. } => &model.introspection,
        }
    }

    // ── MoE Routing ─────────────────────────────────────────────────

    fn set_capture_routing(&self, capture: bool) {
        match self {
            Self::Qwen3Next { model, .. } => model.set_capture_routing(capture),
            Self::Qwen2 { .. } => {} // no MoE in Qwen2
        }
    }

    fn take_routing_data(&self) -> Vec<MoeRoutingData> {
        match self {
            Self::Qwen3Next { model, .. } => model.take_routing_data(),
            Self::Qwen2 { .. } => Vec::new(),
        }
    }

    // ── Activation Patching ─────────────────────────────────────────

    fn set_patch(&self, layer_idx: usize, hidden_state: Tensor) {
        match self {
            Self::Qwen3Next { model, .. } => model.set_patch(layer_idx, hidden_state),
            Self::Qwen2 { model, .. } => {
                let mut intro = model.introspection.lock().unwrap();
                intro.patch_vectors.insert(layer_idx, hidden_state);
            }
        }
    }

    fn clear_patches(&self) {
        match self {
            Self::Qwen3Next { model, .. } => model.clear_patches(),
            Self::Qwen2 { model, .. } => {
                let mut intro = model.introspection.lock().unwrap();
                intro.patch_vectors.clear();
            }
        }
    }

    // ── GDN Recurrent State ─────────────────────────────────────────

    fn gdn_recurrent_states(&self) -> candle_core::Result<Vec<(usize, Tensor)>> {
        match self {
            Self::Qwen3Next { model, .. } => model.gdn_recurrent_states(),
            Self::Qwen2 { .. } => Ok(Vec::new()), // no GDN layers
        }
    }
}

/// High-level model wrapper for introspection experiments.
pub struct IntrospectionModel {
    backend: ModelBackend,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    /// Cached model info computed at load time.
    info: ModelInfo,
}

/// Result of a forward pass with introspection.
pub struct IntrospectionResult {
    /// Model output logits, shape (batch, seq_len, vocab_size).
    pub logits: Tensor,
    /// Hidden states at each layer.
    /// Index 0 = after embedding, index i+1 = after decoder layer i.
    pub hidden_states: Vec<Tensor>,
}

/// Per-layer logit lens probability vectors (softmax over vocab).
pub struct LogitLensResult {
    /// One tensor per layer, shape (batch, vocab_size) — softmax probs at last token position.
    pub layer_probs: Vec<Tensor>,
}

/// Serializable model architecture info.
#[derive(Clone, Serialize)]
pub struct ModelInfo {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub full_attention_interval: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub vocab_size: usize,
    pub layer_types: Vec<String>,
    // GDN-specific dimensions (0 for non-GDN models)
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    // MoE dimensions (0 for non-MoE models)
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
}

/// Used only for auto-detecting model type from config.json.
#[derive(serde::Deserialize)]
struct ConfigProbe {
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    model_type: String,
}

impl IntrospectionModel {
    /// Load a model for introspection from HuggingFace Hub or a local directory.
    ///
    /// `model_path` can be:
    /// - A HuggingFace model ID (e.g., `"Qwen/Qwen2.5-Coder-32B"`)
    /// - An absolute/relative path to a local model directory
    ///
    /// The model type is auto-detected from `config.json`.
    pub fn load(model_path: &str, device: Device, dtype: DType) -> anyhow::Result<Self> {
        let (config_path, weight_files, tokenizer_path) = if Path::new(model_path).is_dir() {
            Self::resolve_local_paths(model_path)?
        } else {
            Self::resolve_hf_paths(model_path)?
        };

        // Probe config.json for model type
        let config_str = std::fs::read_to_string(&config_path)?;
        let probe: ConfigProbe = serde_json::from_str(&config_str)?;

        let is_qwen2 = probe.architectures.iter().any(|a| a == "Qwen2ForCausalLM")
            || probe.model_type == "qwen2";

        let is_qwen3_next = probe.architectures.iter().any(|a| a == "Qwen3NextForCausalLM")
            || probe.model_type == "qwen3_next";

        // Memory-map safetensors files into a sharded VarBuilder
        let vb = unsafe {
            ShardedSafeTensors::sharded(
                &weight_files,
                dtype,
                &device,
                None,
                Arc::new(|_| true),
            )?
        };

        let normal_loading_metadata = NormalLoadingMetadata {
            mapper: Box::new(DummyDeviceMapper {
                nm_device: device.clone(),
            }),
            loading_isq: false,
            real_device: device.clone(),
            multi_progress: Arc::new(MultiProgress::new()),
            matformer_slicing_config: None,
        };

        let (backend, info) = if is_qwen3_next {
            let config: qwen3_next::Config = serde_json::from_str(&config_str)?;
            tracing::info!(
                "Loading Qwen3-Next: {} layers, hidden_size={}, {} experts ({} active)",
                config.num_hidden_layers,
                config.hidden_size,
                config.num_experts,
                config.num_experts_per_tok,
            );

            let model = qwen3_next::Model::new(
                &config,
                vb,
                true,
                normal_loading_metadata,
                AttentionImplementation::Eager,
            )?;

            let layer_types: Vec<String> = (0..config.num_hidden_layers)
                .map(|i| {
                    if (i + 1) % config.full_attention_interval == 0 {
                        "full_attention".to_string()
                    } else {
                        "gdn".to_string()
                    }
                })
                .collect();

            let info = ModelInfo {
                model_type: "qwen3_next".to_string(),
                hidden_size: config.hidden_size,
                num_layers: config.num_hidden_layers,
                num_attention_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                full_attention_interval: config.full_attention_interval,
                num_experts: config.num_experts,
                num_experts_per_tok: config.num_experts_per_tok,
                vocab_size: config.vocab_size,
                layer_types,
                linear_num_key_heads: config.linear_num_key_heads,
                linear_key_head_dim: config.linear_key_head_dim,
                linear_num_value_heads: config.linear_num_value_heads,
                linear_value_head_dim: config.linear_value_head_dim,
                linear_conv_kernel_dim: config.linear_conv_kernel_dim,
                moe_intermediate_size: config.moe_intermediate_size,
                shared_expert_intermediate_size: config.shared_expert_intermediate_size,
            };

            (ModelBackend::Qwen3Next { model, config }, info)
        } else if is_qwen2 {
            let config: qwen2::Config = serde_json::from_str(&config_str)?;
            let head_dim = config.hidden_size / config.num_attention_heads;
            tracing::info!(
                "Loading Qwen2: {} layers, hidden_size={}, head_dim={}",
                config.num_hidden_layers,
                config.hidden_size,
                head_dim,
            );

            let model = qwen2::Model::new(
                &config,
                vb,
                true,
                normal_loading_metadata,
                AttentionImplementation::Eager,
            )?;

            let layer_types: Vec<String> = (0..config.num_hidden_layers)
                .map(|_| "full_attention".to_string())
                .collect();

            let info = ModelInfo {
                model_type: "qwen2".to_string(),
                hidden_size: config.hidden_size,
                num_layers: config.num_hidden_layers,
                num_attention_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim,
                full_attention_interval: 1, // every layer is full attention
                num_experts: 0,
                num_experts_per_tok: 0,
                vocab_size: config.vocab_size,
                layer_types,
                linear_num_key_heads: 0,
                linear_key_head_dim: 0,
                linear_num_value_heads: 0,
                linear_value_head_dim: 0,
                linear_conv_kernel_dim: 0,
                moe_intermediate_size: 0,
                shared_expert_intermediate_size: 0,
            };

            (ModelBackend::Qwen2 { model, config }, info)
        } else {
            anyhow::bail!(
                "Unsupported model architecture: {:?} (model_type: {:?}). \
                 Supported: Qwen3NextForCausalLM, Qwen2ForCausalLM",
                probe.architectures,
                probe.model_type,
            );
        };

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        tracing::info!("Model loaded successfully on {:?}", device);

        Ok(Self {
            backend,
            tokenizer,
            device,
            dtype,
            info,
        })
    }

    /// Get the model's data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    fn resolve_local_paths(dir: &str) -> anyhow::Result<(PathBuf, Vec<PathBuf>, PathBuf)> {
        let dir = Path::new(dir);
        let config_path = dir.join("config.json");
        anyhow::ensure!(
            config_path.exists(),
            "config.json not found in {}",
            dir.display()
        );

        let tokenizer_path = dir.join("tokenizer.json");
        anyhow::ensure!(
            tokenizer_path.exists(),
            "tokenizer.json not found in {}",
            dir.display()
        );

        // Find safetensors weight files
        let index_path = dir.join("model.safetensors.index.json");
        let weight_files = if index_path.exists() {
            let index_str = std::fs::read_to_string(&index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_str)?;
            let weight_map = index["weight_map"]
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("Invalid safetensors index: missing weight_map"))?;
            let filenames: HashSet<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            let mut files: Vec<PathBuf> = filenames.into_iter().map(|f| dir.join(f)).collect();
            files.sort();
            files
        } else {
            let single = dir.join("model.safetensors");
            anyhow::ensure!(
                single.exists(),
                "No safetensors files found in {}",
                dir.display()
            );
            vec![single]
        };

        Ok((config_path, weight_files, tokenizer_path))
    }

    fn resolve_hf_paths(model_id: &str) -> anyhow::Result<(PathBuf, Vec<PathBuf>, PathBuf)> {
        use hf_hub::api::sync::Api;

        tracing::info!("Downloading model from HuggingFace: {}", model_id);

        let api = Api::new()?;
        let repo = api.model(model_id.to_string());

        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        // Check for sharded weights
        let weight_files = match repo.get("model.safetensors.index.json") {
            Ok(index_path) => {
                let index_str = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_str)?;
                let weight_map = index["weight_map"]
                    .as_object()
                    .ok_or_else(|| anyhow::anyhow!("Invalid safetensors index"))?;
                let filenames: HashSet<String> = weight_map
                    .values()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                let mut files = Vec::new();
                for filename in &filenames {
                    tracing::info!("Downloading weight shard: {}", filename);
                    files.push(repo.get(filename)?);
                }
                files.sort();
                files
            }
            Err(_) => vec![repo.get("model.safetensors")?],
        };

        Ok((config_path, weight_files, tokenizer_path))
    }

    /// Run a forward pass capturing hidden states at every layer.
    ///
    /// Not safe to call concurrently — the caller must serialize access.
    pub fn forward_introspect(&self, text: &str) -> anyhow::Result<IntrospectionResult> {
        self.forward_introspect_layers(text, None)
    }

    /// Run a forward pass capturing hidden states at specific layers only.
    ///
    /// `layers` specifies which layers to capture (0 = embedding, 1..N = decoder layers).
    /// Pass `None` to capture all layers. Capturing fewer layers reduces memory copies
    /// during the forward pass — significant on memory-bandwidth-bound systems.
    pub fn forward_introspect_layers(
        &self,
        text: &str,
        layers: Option<std::collections::HashSet<usize>>,
    ) -> anyhow::Result<IntrospectionResult> {
        // Set selective capture before the forward pass
        {
            let mut intro = self.backend.introspection().lock().unwrap();
            intro.capture_layers = layers;
        }

        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids = encoding.get_ids();

        let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;

        let flash_params = FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: HashMap::new(),
            cumulative_seqlens_k: HashMap::new(),
            causal: true,
        };

        let seq_len = input_ids.len();
        let (logits, hidden_states) = self.backend.forward_introspect(
            &input_tensor,
            &[0],
            vec![(0, seq_len)],
            None,
            &flash_params,
        )?;

        // Reset capture_layers
        {
            let mut intro = self.backend.introspection().lock().unwrap();
            intro.capture_layers = None;
        }

        Ok(IntrospectionResult {
            logits,
            hidden_states,
        })
    }

    /// Run logit lens on hidden states from [`forward_introspect`].
    ///
    /// Returns per-layer softmax probability vectors at the last token position.
    pub fn logit_lens(&self, hidden_states: &[Tensor]) -> anyhow::Result<LogitLensResult> {
        let layer_probs = self.backend.logit_lens_all(hidden_states)?;
        Ok(LogitLensResult { layer_probs })
    }

    /// Tokenize text and return (token_ids, token_strings).
    pub fn tokenize(&self, text: &str) -> anyhow::Result<(Vec<u32>, Vec<String>)> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let ids = encoding.get_ids().to_vec();
        let tokens: Vec<String> = ids
            .iter()
            .map(|&id| {
                self.tokenizer
                    .decode(&[id], false)
                    .unwrap_or_else(|_| format!("<{}>", id))
            })
            .collect();
        Ok((ids, tokens))
    }

    /// Decode a single token ID to its string representation.
    pub fn decode_token(&self, id: u32) -> String {
        self.tokenizer
            .decode(&[id], false)
            .unwrap_or_else(|_| format!("<{}>", id))
    }

    /// Set a steering vector for a specific layer.
    pub fn set_steering_vector(&self, layer_idx: usize, vector: Tensor) {
        self.backend.set_steering_vector(layer_idx, vector);
    }

    /// Set the same steering vector (scaled) across a range of layers.
    pub fn set_steering_vectors_range(
        &self,
        layers: std::ops::Range<usize>,
        vector: &Tensor,
        scale: f64,
    ) -> anyhow::Result<()> {
        self.backend
            .set_steering_vectors_range(layers, vector, scale)?;
        Ok(())
    }

    /// Clear all steering vectors.
    pub fn clear_steering_vectors(&self) {
        self.backend.clear_steering_vectors();
    }

    /// Get model architecture information.
    pub fn model_info(&self) -> ModelInfo {
        self.info.clone()
    }

    /// Get a reference to the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get the device the model is loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Resolve stop token IDs from the tokenizer.
    ///
    /// Tries `<|im_end|>` and `<|endoftext|>` — the standard Qwen stop tokens.
    fn stop_token_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        for tok_str in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(id) = self.tokenizer.token_to_id(tok_str) {
                ids.push(id);
            }
        }
        ids
    }

    /// Generate text autoregressively with steering vectors active.
    ///
    /// Prefills the KV cache with the prompt, then decodes one token at a time.
    /// Steering vectors (if set) remain active throughout generation.
    /// Returns the generated token string (excluding the prompt).
    pub fn generate(
        &self,
        text: &str,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
    ) -> anyhow::Result<GenerationResult> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
        let input_ids = encoding.get_ids().to_vec();
        let prompt_len = input_ids.len();
        let stop_tokens = self.stop_token_ids();

        let flash_params = FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: HashMap::new(),
            cumulative_seqlens_k: HashMap::new(),
            causal: true,
        };

        // Prefill: process entire prompt, extract only last position logits
        let input_tensor = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;
        let logits = self.backend.forward(
            &input_tensor,
            &[0],
            vec![(prompt_len - 1, 1)],
            None,
            &flash_params,
        )?;

        let mut generated_ids = Vec::new();
        let mut next_token = sample_token(&logits, temperature, top_p)?;
        generated_ids.push(next_token);

        if stop_tokens.contains(&next_token) {
            return Ok(GenerationResult {
                text: String::new(),
                token_ids: generated_ids,
                stop_reason: StopReason::StopToken,
            });
        }

        // Decode loop
        for step in 0..max_new_tokens.saturating_sub(1) {
            let offset = prompt_len + step;
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.backend.forward(
                &input,
                &[offset],
                vec![(0, 1)],
                None,
                &flash_params,
            )?;

            next_token = sample_token(&logits, temperature, top_p)?;
            generated_ids.push(next_token);

            if stop_tokens.contains(&next_token) {
                break;
            }
        }

        let stop_reason = if generated_ids
            .last()
            .map_or(false, |t| stop_tokens.contains(t))
        {
            StopReason::StopToken
        } else {
            StopReason::MaxTokens
        };

        // Decode generated tokens (exclude stop token from text)
        let decode_ids: Vec<u32> = if stop_reason == StopReason::StopToken {
            generated_ids[..generated_ids.len() - 1].to_vec()
        } else {
            generated_ids.clone()
        };

        let text = self
            .tokenizer
            .decode(&decode_ids, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;

        Ok(GenerationResult {
            text,
            token_ids: generated_ids,
            stop_reason,
        })
    }

    /// Compute concept activation scores from hidden states.
    ///
    /// For each requested layer, computes the dot product between the last-token
    /// hidden state and the concept direction vector. Returns per-layer activation
    /// scores — higher means the model's internal representation is more aligned
    /// with the concept.
    pub fn concept_activation(
        &self,
        hidden_states: &[Tensor],
        concept_vectors: &HashMap<usize, Vec<f32>>,
    ) -> anyhow::Result<HashMap<usize, f32>> {
        use candle_core::IndexOp;

        let mut activations = HashMap::new();

        for (&layer_idx, direction) in concept_vectors {
            if layer_idx >= hidden_states.len() {
                continue;
            }
            let hs = &hidden_states[layer_idx]; // (1, seq_len, hidden_size)
            let seq_len = hs.dim(1)?;
            let last = hs
                .i((0, seq_len - 1))?
                .to_dtype(DType::F32)?; // (hidden_size,)

            let cv = Tensor::new(direction.as_slice(), &self.device)?
                .to_dtype(DType::F32)?;
            let dot = (&last * &cv)?.sum_all()?.to_scalar::<f32>()?;
            activations.insert(layer_idx, dot);
        }

        Ok(activations)
    }

    // ── MoE Routing Capture ─────────────────────────────────────────

    /// Enable or disable MoE routing capture for subsequent forward passes.
    pub fn set_capture_routing(&self, capture: bool) {
        self.backend.set_capture_routing(capture);
    }

    /// Take captured MoE routing data, leaving the buffer empty.
    pub fn take_routing_data(&self) -> Vec<MoeRoutingData> {
        self.backend.take_routing_data()
    }

    /// Run a forward pass with routing capture enabled.
    /// Returns (logits, hidden_states, routing_data).
    pub fn forward_introspect_with_routing(
        &self,
        text: &str,
    ) -> anyhow::Result<(Tensor, Vec<Tensor>, Vec<MoeRoutingData>)> {
        self.set_capture_routing(true);
        let result = self.forward_introspect(text)?;
        let routing = self.take_routing_data();
        self.set_capture_routing(false);
        Ok((result.logits, result.hidden_states, routing))
    }

    // ── Activation Patching ─────────────────────────────────────────

    /// Set a patch vector that REPLACES the hidden state at the given layer.
    pub fn set_patch(&self, layer_idx: usize, hidden_state: Tensor) {
        self.backend.set_patch(layer_idx, hidden_state);
    }

    /// Clear all activation patches.
    pub fn clear_patches(&self) {
        self.backend.clear_patches();
    }

    // ── GDN Recurrent State ─────────────────────────────────────────

    /// Extract GDN recurrent states from the cache after a forward pass.
    /// Returns (layer_idx, recurrent_state) for each GDN layer.
    /// Empty on non-GDN models.
    pub fn gdn_recurrent_states(&self) -> anyhow::Result<Vec<(usize, Tensor)>> {
        Ok(self.backend.gdn_recurrent_states()?)
    }
}

/// Result of autoregressive generation.
pub struct GenerationResult {
    /// Generated text (excluding prompt and stop token).
    pub text: String,
    /// All generated token IDs (including stop token if hit).
    pub token_ids: Vec<u32>,
    /// Why generation stopped.
    pub stop_reason: StopReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    MaxTokens,
    StopToken,
}

/// Sample a single token from logits using temperature + optional top-p.
fn sample_token(
    logits: &Tensor,
    temperature: f64,
    top_p: Option<f64>,
) -> anyhow::Result<u32> {
    use candle_core::IndexOp;
    use rand::distr::Distribution;

    // logits shape: (1, extracted_len, vocab_size) — we want the last position
    let seq_len = logits.dim(1)?;
    let logits_1d = logits
        .i((0, seq_len - 1))?
        .to_dtype(DType::F32)?; // (vocab_size,)

    if temperature < 1e-7 {
        // Greedy
        let argmax = logits_1d.argmax(0)?;
        return Ok(argmax.to_scalar::<u32>()?);
    }

    // Temperature scaling + softmax
    let scaled = (&logits_1d / temperature)?;
    let max_val = scaled.max(0)?;
    let shifted = scaled.broadcast_sub(&max_val)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_all()?;
    let probs = exp.broadcast_div(&sum)?;
    let mut probs_vec: Vec<f32> = probs.to_vec1()?;

    // Clamp NaN/Inf/negative to 0 (can happen with aggressive steering)
    for p in probs_vec.iter_mut() {
        if !p.is_finite() || *p < 0.0 {
            *p = 0.0;
        }
    }
    // Ensure at least one non-zero probability
    if probs_vec.iter().all(|&p| p == 0.0) {
        // Fall back to argmax on original logits
        let argmax = logits_1d.argmax(0)?;
        return Ok(argmax.to_scalar::<u32>()?);
    }

    // Top-p nucleus sampling
    if let Some(p) = top_p {
        apply_top_p(&mut probs_vec, p as f32);
    }

    let dist = rand::distr::weighted::WeightedIndex::new(&probs_vec)
        .map_err(|e| anyhow::anyhow!("Sampling error: {}", e))?;
    let mut rng = rand::rng();
    Ok(dist.sample(&mut rng) as u32)
}

/// Zero out tokens outside the top-p nucleus and renormalize in place.
fn apply_top_p(probs: &mut Vec<f32>, p: f32) {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, prob)) in indexed.iter().enumerate() {
        cumulative += prob;
        if cumulative > p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Build set of kept indices
    let kept: HashSet<usize> = indexed.iter().take(cutoff_idx).map(|&(i, _)| i).collect();
    let mut sum = 0.0;
    for (i, v) in probs.iter_mut().enumerate() {
        if !kept.contains(&i) {
            *v = 0.0;
        } else {
            sum += *v;
        }
    }
    if sum > 0.0 {
        for v in probs.iter_mut() {
            *v /= sum;
        }
    }
}
