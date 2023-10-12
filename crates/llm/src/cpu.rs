#![allow(unused_imports)]
use tracing::{error, info};
use std::{
    collections::HashMap,
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use async_trait::async_trait;
// #[cfg(feature = "ee")]
//pub use crate::embedder::*;

///
#[async_trait]
pub trait Embedder: Send + Sync {
    ///
    async fn embed(&self, data: &str) -> anyhow::Result<Embedding>;
    //fn tokenizer(&self) -> &Tokenizer;
    ///
    async fn batch_embed(&self, log: Vec<&str>) -> anyhow::Result<Vec<Embedding>>;
}

///
pub struct LocalEmbedder {
    model: Box<dyn crate::Model>,
    sessions: Vec<Arc<tokio::sync::Mutex<crate::InferenceSession>>>,
    //tokenizer: Tokenizer,
    permits: Arc<tokio::sync::Semaphore>,
}

// InferenceSession is explicitly not Sync because it uses ggml::Tensor internally,
// Bert does not make use of these tensors however
unsafe impl Sync for LocalEmbedder {}

impl LocalEmbedder {
    ///
    pub fn new(model_path: &Path) -> anyhow::Result<Self> {
        //let mut model_params = crate::ModelParameters::default();
        //model_params.use_gpu = true;
        // Load model
        let model_params = crate::ModelParameters {
            prefer_mmap: true,
            //context_size: 2048,
            use_gpu: false,
            //gpu_layers: 10,
            ..Default::default()
        };

        let model_architecture = crate::ModelArchitecture::ClipVision;
        let model = crate::load_dynamic(
            Some(model_architecture),
            //&model_dir.join("ggml").join("ggml-model-q4_0.bin"),
            model_path,
            // this tokenizer is used for embedding
            /*crate::TokenizerSource::HuggingFaceTokenizerFile(
                model_dir.join("ggml").join("tokenizer.json"),
            ),*/
            crate::TokenizerSource::Embedded,
            model_params,
            crate::load_progress_callback_stdout,
        )?;

        // TODO: this can be parameterized
        //
        // the lower this number, the more time we might spend waiting to run an embedding.
        // the higher this number, the more vram we use, currently we use ~2G per session. this
        // can be fixed by disabling scratch buffers in ggml, bert has no use for this.
        let session_count = 1;
        info!(%session_count, "spawned inference sessions");

        let sessions = (0..session_count)
            .map(|_| {
                model.start_session(crate::InferenceSessionConfig {
                    ..Default::default()
                })
            })
            .map(tokio::sync::Mutex::new)
            .map(Arc::new)
            .collect();

        // this tokenizer is used for chunking - do not pad or truncate chunks
        //let mut tokenizer = Tokenizer::from_file(model_dir.join("ggml").join("tokenizer.json")).unwrap();
        //let _ = tokenizer.with_padding(None).with_truncation(None);

        Ok(Self {
            model,
            sessions,
            //tokenizer,
            permits: Arc::new(tokio::sync::Semaphore::new(session_count)),
        })
    }
}

///
pub type Embedding = Vec<f32>;

///
pub fn get_image_token_ids(image_paths: &[&str]) -> Vec<Vec<u32>> {
    let images: Vec<Vec<u32>> = image_paths
        .iter()
        .map(|image_path| {
            let image = image::open(&image_path).unwrap().to_rgb8();
            let resized =
                image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
            //println!("{:#?}", resized.dimensions());
            let img_vec_resized: Vec<u32> = resized.to_vec().iter().map(|&i| i as u32).collect();
            assert_eq!(img_vec_resized.len(), 1 * 3 * 224 * 224);
            println!("{:?}: \n{:?}", image_path, img_vec_resized.get(0..10));
            img_vec_resized
        })
        .collect();
    //let images: Vec<_> = images.iter().map(AsRef::as_ref).collect();
    images
}

#[async_trait]
impl Embedder for LocalEmbedder {
    async fn embed(&self, image_path: &str) -> anyhow::Result<Embedding> {
        let mut output_request = crate::OutputRequest {
            all_logits: None,
            embeddings: Some(Vec::new()),
        };
        /*let vocab = self.model.tokenizer();
        let beginning_of_sentence = true;
        let query_token_ids = vocab
            .tokenize(sequence, beginning_of_sentence)
            .unwrap()
            .iter()
            .map(|(_, tok)| *tok)
            .collect::<Vec<_>>();*/

        let input_token_ids = get_image_token_ids(&[image_path]);
        let input_token_ids = &input_token_ids[0];

        if let Ok(_permit) = self.permits.acquire().await {
            for s in &self.sessions {
                if let Ok(mut session) = s.try_lock() {
                    self.model
                        .evaluate(&mut session, &input_token_ids, &mut output_request);
                    return Ok(output_request.embeddings.unwrap());
                }
            }
        }

        unreachable!();
    }

    /*fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }*/

    async fn batch_embed(&self, image_paths: Vec<&str>) -> anyhow::Result<Vec<Embedding>> {
        let mut output_request = crate::OutputRequest {
            all_logits: None,
            embeddings: Some(Vec::new()),
        };
        //let vocab = self.model.tokenizer();
        //let beginning_of_sentence = true;
        let input_token_ids = get_image_token_ids(&image_paths);
        let input_token_ids: Vec<_> = input_token_ids.iter().map(AsRef::as_ref).collect();

        if let Ok(_permit) = self.permits.acquire().await {
            for s in &self.sessions {
                if let Ok(mut session) = s.try_lock() {
                    let embeddings = self.model.batch_evaluate(
                        &mut session,
                        &input_token_ids,
                        &mut output_request,
                    ).unwrap();
                    return Ok(embeddings);
                }
            }
        }
        unreachable!()
    }
}


// reference from https://github.com/BloopAI/bloop/pull/932/files
// Queue
/*#[derive(Default)]
pub struct EmbedQueue {
    log: scc::Queue<Mutex<Option<EmbedChunk>>>,
    len: AtomicUsize,
}
impl EmbedQueue {
    pub fn pop(&self) -> Option<EmbedChunk> {
        let Some(val) = self.log.pop()
	else {
	    return None;
	};
        // wrapping shouldn't happen, because only decrements when
        // `log` is non-empty.
        self.len.fetch_sub(1, Ordering::SeqCst);
        let val = val.lock().unwrap().take().unwrap();
        Some(val)
    }
    pub fn push(&self, chunk: EmbedChunk) {
        self.log.push(Mutex::new(Some(chunk)));
        self.len.fetch_add(1, Ordering::SeqCst);
    }
    pub fn len(&self) -> usize {
        self.len.load(Ordering::SeqCst)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
#[derive(Default)]
pub struct EmbedChunk {
    pub id: String,
    pub data: String,
    pub payload: HashMap<String, qdrant_client::qdrant::Value>,
}*/