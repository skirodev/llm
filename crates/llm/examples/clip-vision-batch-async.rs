#![allow(unused_variables, dead_code, unused_imports)]
use std::path::PathBuf;
use std::sync::Arc;
use llm::*;

#[tokio::main]
async fn main() {
    let model_path = PathBuf::from(
        r"C:\Users\11048\Documents\cpp\chatglm.cpp\CLIP-ViT-B-32-laion2B-s34B-b79K\ggml-model-f16.bin",
        //r"C:\Users\11048\Documents\cpp\chatglm.cpp\clip-vit-base-patch32\ggml-model-f16.bin",
        //r"C:\Users\11048\Documents\cpp\chatglm.cpp\chinese-clip-vit-large-patch14/ggml-model-f16.bin",
        //r"C:\Users\11048\Documents\cpp\chatglm.cpp\QA-CLIP-ViT-B-16\ggml-model-f16.bin",
        //r"C:\Users\11048\Documents\cpp\chatglm.cpp\chinese-clip-vit-base-patch16\ggml-model-f16.bin",
    );
    //let img_path = r"C:\Users\11048\Pictures\images\pixabay.com\set1\white-flower-7990645_1280.jpg";
    let image_paths = vec![
        //bin/image-search-build -m /home/skiro/code/ai/models/laion_clip-vit-b-32-laion2b-s34b-b79k.ggmlv0.q4_1.bin /home/skiro/code/ai/images
        r"C:\Users\11048\Pictures\images\pixabay.com\set1\white-flower-7990645_1280.jpg",
        //r"C:\Users\11048\Pictures\images\pixabay.com\set1\sun-8066051_1280.jpg",
        //r"C:\Users\11048\Pictures\images\pixabay.com\set1\sun-8066051_1280.jpg",
        //r"C:\Users\11048\Pictures\images\pixabay.com\set1\white-flower-7990645_1280.jpg",
        //r"C:\Users\11048\Pictures\00148-459525319.png",
        //r"C:\Users\11048\Pictures\images\pixabay.com\set1\white-flower-7990645_1280.jpg",
        //r"C:\Users\11048\Pictures\00148-459525319.png",
        //r"C:\Users\11048\Pictures\images\pixabay.com\set1\white-flower-7990645_1280.jpg",
    ];
    //let text = "a";
    //let text = "a sitting man";
    let text = "girl";

    let embeder = Arc::new(LocalEmbedder::new(&model_path).unwrap());
    let batch_vision_embeddings: Vec<Vec<f32>> = embeder.batch_embed(image_paths.clone()).await.unwrap();
    /*.chunks(512)
    //.chunks(768)
    .map(|chunk| chunk.to_vec())
    .collect();*/
    //let text_embeddings = get_embeddings(text_model.as_ref(), &inference_parameters, text);
    //save_vec(&vision_embeddings);

    // Print embeddings
    fn print_embeddings(text: &str, embeddings: &[f32]) {
        println!("{text}");
        println!("  Embeddings length: {}", embeddings.len());
        println!("  Embeddings first 10: {:.06?}", embeddings.get(0..10));
        /*if embeddings.len() > 512 {
            println!(
                "  Embeddings first 512-522: {:.06?}",
                embeddings.get(512..522)
            );
        }
        if embeddings.len() > 1024 {
            println!(
                "  Embeddings first 1024-1044: {:.06?}",
                embeddings.get(1024..1034)
            );
        }*/
    }

    //batch_vision_embeddings.iter().map(|vision_embeddings| {})
    for (filepath, vision_embedding) in std::iter::zip(image_paths, batch_vision_embeddings) {
        print_embeddings(filepath, &vision_embedding);
    }
    //print_embeddings(text, &text_embeddings);

    //let similarity = cosine_similarity(&vision_embeddings, &text_embeddings);
    //println!("Similarity: {:?}", similarity);
}

/*fn get_embeddings(
    model: &dyn llm::Model,
    inference_parameters: &llm::InferenceParameters,
    query: &str,
) -> Vec<f32> {
    let mut session = model.start_session(Default::default());
    let mut output_request = llm::OutputRequest {
        all_logits: None,
        embeddings: Vec::new(),
    };
    /*let vocab = model.tokenizer();
    let beginning_of_sentence = true;
    let query_token_ids = vocab
        .tokenize(query, beginning_of_sentence)
        .unwrap()
        .iter()
        .map(|(_, tok)| *tok)
        .collect::<Vec<_>>();*/
    let query_token_ids = clip_tokenize(model, query);
    model.evaluate(&mut session, &query_token_ids, &mut output_request);
    output_request.embeddings.unwrap()
}*/

fn get_image_embeddings(
    model: &dyn llm::Model,
    _inference_parameters: &llm::InferenceParameters,
    //image_path: &str,
    image_paths: &[&str],
) -> Vec<Vec<f32>> {
    let mut session = model.start_session(llm::InferenceSessionConfig {
        memory_k_type: llm::ModelKVMemoryType::Float16,
        memory_v_type: llm::ModelKVMemoryType::Float16,
        n_batch: 512,
        n_threads: 8,
    });
    let mut output_request = llm::OutputRequest {
        all_logits: None,
        embeddings: Some(Vec::new()),
    };
    //single
    /*let image = image::open(&image_path).unwrap().to_rgb8();
    //let image: Vec<u8> = image::open(image_path).unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    //println!("{:#?}", resized.dimensions());
    let img_vec_resized: Vec<u32> = resized.to_vec().iter().map(|&i| i as u32).collect();*/
    // batch
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
    let images: Vec<_> = images.iter().map(AsRef::as_ref).collect();
    //let flattened_imgs = imgs.concat();
    model
        .batch_evaluate(&mut session, &images, &mut output_request)
        .unwrap()
}

// chinese clip
fn clip_tokenize(_model: &dyn llm::Model, text: &str) -> Vec<llm::TokenId> {
    use tokenizers::models::wordpiece::{WordPiece, WordPieceTrainerBuilder};
    use tokenizers::normalizers::{BertNormalizer, NormalizerWrapper};
    use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
    use tokenizers::processors::bert::BertProcessing;
    use tokenizers::{decoders, EncodeInput, Model, TokenizerImpl};

    use tokenizers::decoders::DecoderWrapper;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::processors::PostProcessorWrapper;

    type BertTokenizer = TokenizerImpl<
        WordPiece,
        BertNormalizer,
        BertPreTokenizer,
        BertProcessing,
        decoders::wordpiece::WordPiece,
    >;

    /// Resembling the BertTokenizer implementation from the Python bindings.
    fn create_bert_tokenizer(wp: WordPiece) -> BertTokenizer {
        let sep_id = *wp.get_vocab().get("[SEP]").unwrap();
        let cls_id = *wp.get_vocab().get("[CLS]").unwrap();
        let mut tokenizer = TokenizerImpl::new(wp);
        tokenizer.with_pre_tokenizer(BertPreTokenizer);
        tokenizer.with_normalizer(BertNormalizer::default());
        tokenizer.with_decoder(decoders::wordpiece::WordPiece::default());
        tokenizer.with_post_processor(BertProcessing::new(
            ("[SEP]".to_string(), sep_id),
            ("[CLS]".to_string(), cls_id),
        ));
        tokenizer
    }

    let wp = WordPiece::from_file(
        r"C:\Users\11048\Documents\cpp\chatglm.cpp\chinese-clip-vit-base-patch16\vocab.txt",
    )
    .build()
    .unwrap();
    let tokenizer = create_bert_tokenizer(wp);

    let tokens: tokenizers::Encoding = tokenizer.encode(text, true).unwrap();

    tokens.get_ids().into()
}

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product = dot(v1, v2);
    let magnitude1 = magnitude(v1);
    let magnitude2 = magnitude(v2);

    dot_product / (magnitude1 * magnitude2)
}

fn dot(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(&x, &y)| x * y).sum()
}

fn magnitude(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}
