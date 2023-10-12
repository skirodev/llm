//! An implementation of [GPT-NeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox) for the `llm` ecosystem.
//! This crate also supports the [RedPajama](https://www.together.xyz/blog/redpajama) GPT-NeoX model.
#![allow(unreachable_code, unused_variables, unused_mut, dead_code, unused_imports)]
#![deny(missing_docs)]

use std::{error::Error, sync::Arc};

use ggml::Tensor;
use llm_base::{
    ggml,
    model::{common, HyperparametersWriteError},
    util, FileType, GraphOutputs, InferenceSession, InferenceSessionConfig, KnownModel, LoadError, ModelContext,
    ModelParameters, OutputRequest, Regex, TensorLoader, TokenId, Tokenizer,
};

/// The GPT-NeoX model. Ref: [GitHub](https://github.com/EleutherAI/gpt-neox)
///
/// # Safety
/// This implements [Send] and [Sync] as it is immutable after construction.
pub struct ClipVision {
    params: ModelParameters,

    hyperparameters: Hyperparameters,
    tokenizer: Tokenizer,

    // model-global weights
    // normalization gain & bias
    vision_class_embedding: Tensor,
    vision_patch_embeddings: Tensor,
    vision_position_embeddings: Tensor,
    // weight token embeddings
    pre_ln_w: Tensor,
    pre_ln_b: Tensor,
    post_ln_w: Tensor,
    post_ln_b: Tensor,
    // language model head gain
    projection: Tensor,

    // weights for the model
    layers: Vec<Layer>,

    // must be kept alive for the model
    context: ModelContext,
}

unsafe impl Send for ClipVision {}
unsafe impl Sync for ClipVision {}

impl KnownModel for ClipVision {
    type Hyperparameters = Hyperparameters;

    fn new<E: Error>(
        hyperparameters: Hyperparameters,
        params: ModelParameters,
        tokenizer: Tokenizer,
        tensor_loader: impl TensorLoader<E>,
    ) -> Result<Self, E>
    where
        Self: Sized,
    {
        let mut tl = tensor_loader;
        let backend = params.backend(0);

        // model-global weights
        let vision_class_embedding = tl.load("vision_model.embeddings.class_embedding")?;
        let vision_patch_embeddings = tl.load("vision_model.embeddings.patch_embedding.weight")?;
        let vision_position_embeddings = tl.load("vision_model.embeddings.position_embedding.weight")?;
        let pre_ln_w = tl.load("vision_model.pre_layrnorm.weight")?;
        let pre_ln_b = tl.load("vision_model.pre_layrnorm.bias")?;

        let post_ln_w = tl.load("vision_model.post_layernorm.weight")?.transfer_to(backend);
        let post_ln_b = tl.load("vision_model.post_layernorm.bias")?.transfer_to(backend);
        let projection = tl.load("visual_projection.weight")?.transfer_to(backend);

        let mut layers = Vec::new();
        for i in 0..hyperparameters.v_n_layer {
            let backend = params.backend(i);

            let layer = Layer {
                // vision
                q_w: tl.load(&format!(
                    "vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
                ))?.transfer_to(backend),
                q_b: tl.load(&format!(
                    "vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
                ))?.transfer_to(backend),
                k_w: tl.load(&format!(
                    "vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
                ))?.transfer_to(backend),
                k_b: tl.load(&format!(
                    "vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
                ))?.transfer_to(backend),
                v_w: tl.load(&format!(
                    "vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
                ))?.transfer_to(backend),
                v_b: tl.load(&format!(
                    "vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
                ))?.transfer_to(backend),
                o_w: tl.load(&format!(
                    "vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
                ))?.transfer_to(backend),
                o_b: tl.load(&format!(
                    "vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
                ))?.transfer_to(backend),

                ln_1_w: tl.load(&format!(
                    "vision_model.encoder.layers.{i}.layer_norm1.weight"
                ))?.transfer_to(backend),
                ln_1_b: tl.load(&format!("vision_model.encoder.layers.{i}.layer_norm1.bias"))?.transfer_to(backend),

                ff_i_w: tl.load(&format!("vision_model.encoder.layers.{i}.mlp.fc1.weight"))?.transfer_to(backend),
                ff_i_b: tl.load(&format!("vision_model.encoder.layers.{i}.mlp.fc1.bias"))?.transfer_to(backend),
                ff_o_w: tl.load(&format!("vision_model.encoder.layers.{i}.mlp.fc2.weight"))?.transfer_to(backend),
                ff_o_b: tl.load(&format!("vision_model.encoder.layers.{i}.mlp.fc2.bias"))?.transfer_to(backend),

                ln_2_w: tl.load(&format!(
                    "vision_model.encoder.layers.{i}.layer_norm2.weight"
                ))?.transfer_to(backend),
                ln_2_b: tl.load(&format!("vision_model.encoder.layers.{i}.layer_norm2.bias"))?.transfer_to(backend),
            };

            layers.push(layer);
        }

        let context = tl.finish();

        Ok(ClipVision {
            hyperparameters,
            params,
            tokenizer,
            vision_class_embedding,
            vision_patch_embeddings,
            vision_position_embeddings,
            pre_ln_w,
            pre_ln_b,
            post_ln_w,
            post_ln_b,
            projection,
            layers,
            context: context,
        })
    }

    fn start_session(&self, config: InferenceSessionConfig) -> InferenceSession {
        InferenceSession::new(
            config,
            &self.params,
            self.hyperparameters.v_n_layer,
            self.hyperparameters.v_n_embd,
            self.hyperparameters.t_n_vocab,
        )
    }

    // allow snake case here as its a one-to-one mapping of the original names
    #[tracing::instrument(level = "trace", skip_all)]
    #[allow(non_snake_case)]
    fn evaluate(
        &self,
        session: &mut InferenceSession,
        //input_tokens: &[TokenId],
        input_tokens: &[u32],
        output_request: &mut OutputRequest,
    ) {
        let IMAGE_BATCH_SIZE: usize = input_tokens.clone().len() / (1 * 3 * 224 * 224);
        assert_eq!(input_tokens.len() % (1 * 3 * 224 * 224), 0);
        println!("Get Images: {:?}", input_tokens.len() / (1 * 3 * 224 * 224));
        //let flattened_imgs = imgs.concat().as_slice();
        /*let imgs: Vec<Vec<u32>> = flattened_imgs.chunks(3)
        .map(|chunk| chunk.to_vec())
        .collect();*/
        //let n = input_tokens.len();
        let n_past = session.n_past;
        let n_ctx = self.params.context_size;

        let Hyperparameters {
            // text
            t_n_vocab,
            t_n_pos,
            t_n_embd,
            t_n_intermediate,
            t_projection_dim,
            t_n_head,
            t_n_layer,
            // vision
            v_image_size,
            v_patch_size,
            v_n_embd,
            v_n_intermediate,
            v_projection_dim,
            v_n_head,
            v_n_layer,
            // others
            use_gelu,
            ..
        } = self.hyperparameters;

        let outputs = session.compute(self.context.clone(), input_tokens, |builder| {
            let mut imgs: Vec<Vec<u32>> = input_tokens.chunks(1 * 3 * 224 * 224)
                .map(|chunk| chunk.to_vec())
                .collect();
            assert_eq!(imgs.len(), IMAGE_BATCH_SIZE);

            let mut ctx0 = builder.ctx0.borrow_mut();
            // Do not use embd
            //let embd = builder.embd;
            //let mut input_layer = ctx0.op_get_rows(&self.wte, embd);
            let mut input_layer: Tensor;
            let (memory_k_size, memory_v_size) = (
                builder.memory_k.element_size(),
                builder.memory_v.element_size(),
            );

            let num_patches = (v_image_size / v_patch_size) * (v_image_size / v_patch_size);
            let num_positions = num_patches + 1;
            let v_head_dim = v_n_embd / v_n_head;
            //int batch_size = imgs.size();
            let batch_size = imgs.len();

            let mut gf = ctx0.create_compute_graph();
            //std::process::exit(0);

            // open image, resize it and make a Tensor out of it
            /*let image = image::open(r"\\wsl.localhost\Ubuntu-22.04\home\skiro\code\ai\images\test1.png").unwrap().to_rgb8();
            //let image: Vec<u8> = image::open(image_path).unwrap().to_rgb8();
            let resized =
                image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
            println!("{:#?}", resized.dimensions());*/
            /*let image: Tensor =
            tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
                let mean = [0.48145466, 0.4578275, 0.40821073][c];
                let std = [0.26862954, 0.26130258, 0.27577711][c];
                (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
            })
            .into();*/
            let mut inp_raw = ctx0.new_tensor_4d(
                ggml::Type::F32,
                //batch_size, 3, 224, 224,
                224,
                224,
                3,
                //1
                batch_size,
            );
            //println!("Debug 21");
            /*let img_arr_u8: ndarray::Array4<u8> = ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
                resized[(x as _, y as _)][c]
            }).into();*/
            //let img_vec_resized: Vec<u8> = resized.to_vec();
            //let img_vec_u8: Vec<u8> = img_arr_u8.iter().cloned().collect();
            //let 
            //assert_eq!(img_vec_resized, img_vec_u8);
            let imgs_arr: Vec<ndarray::Array4<f32>> = imgs.iter().map(|img| {
                println!("{:?}", img.get(0..10));
                ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
                    let mean = [0.48145466, 0.4578275, 0.40821073][c];
                    let std = [0.26862954, 0.26130258, 0.27577711][c];
                    /*
                    bin/image-search-build -m /workspaces/clip.cpp/build/laion_clip-vit-b-32-laion2b-s34b-b79k.ggmlv0.q4_1.bin /workspaces/clip.cpp/tests
                    clip_model_load: loading model from '/workspaces/clip.cpp/build/laion_clip-vit-b-32-laion2b-s34b-b79k.ggmlv0.q4_1.bin' - please wait....................................................clip_model_load: model size =    93.92 MB / num tensors = 397
                    clip_model_load: model loaded

                    main: starting base dir scan of '/workspaces/clip.cpp/tests'

                    main: processing 2 files in 'tests'
                    .Segmentation fault (core dumped)
                    */
                    //assert_eq!(img_vec_resized[3 * (y * 224 + x) as usize + c], resized[(x as _, y as _)][c]);
                    //(resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
                    //println!("{:?}", (img[3 * (y * 224 + x) as usize + c] as f32 / 255.0 - mean) / std);
                    (img[3 * (y * 224 + x) as usize + c] as f32 / 255.0 - mean) / std
    
    
                    //resized[(x as _, y as _)][c] as f32
                    //(x*1000+y*100+c*10) as f32
                })
                .into()
            }).collect::<Vec<ndarray::Array4<f32>>>();
            /*let imgs_arr: Vec<ndarray::Array4<f32>> = imgs.into_iter()
            .map(|img_vec_resized| {
                ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)| {
                let mean = [0.48145466, 0.4578275, 0.40821073][c];
                let std = [0.26862954, 0.26130258, 0.27577711][c];
                //assert_eq!(img_vec_resized[3 * (y * 224 + x) as usize + c], resized[(x as _, y as _)][c]);
                //(resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
                (img_vec_resized[3 * (y * 224 + x) as usize + c] as f32 / 255.0 - mean) / std


                //resized[(x as _, y as _)][c] as f32
                //(x*1000+y*100+c*10) as f32
            })
            .into();
            })
            .collect();*/
            assert_eq!(imgs_arr.len(), IMAGE_BATCH_SIZE);
            let mut data: Vec<f32> = imgs_arr.iter().map(
                |mut img_arr| {
                    let mut a = img_arr.iter().cloned().collect::<Vec<f32>>();
                    println!("Inner: {:?}", a.get(0..20));
                    //a.reverse();
                    a
                }
            ).collect::<Vec<Vec<f32>>>().concat();
            /*for i in 0..150528 {
                //assert_eq!(data[i], data[i+150528]);
                if data[i] != data[i+150528] {
                    println!("Got Error: \nIndex {:?}: {:?}\nIndex {:?}: {:?}", i, data[i], i+150528, data[i+150528]);
                    std::process::exit(0);
                }
            }*/
            println!("{:?}\n{:?}", data.get(0..20), data.get(150528..150528+20));
            //data.reverse();
            assert_eq!(data.len(), 1 * 3 * 224 * 224 * IMAGE_BATCH_SIZE);
            /*let data: Vec<Vec<f32>> = imgs_arr.iter().map(|img_arr| img_arr.iter().cloned().collect()
            ).collect();*/
            /*let mut data = vec![0f32; batch_size * 3 * 224 * 224];
            let (nx, ny) = (224u32, 224u32);
            let nn = (nx * ny) as usize;
            for b in 0..batch_size {
                for c in 0..3 {
                    for y in 0..ny {
                        for x in 0..nx {
                            let img_vec_idx = 3 * (y * nx + x) as usize + c;
                            let idx = b * 3 * nn + c * nn + (y * nx) as usize + x as usize;
                            //data[idx] = resized[(x, y)][c];

                            /*let mean = [0.48145466, 0.4578275, 0.40821073][c];
                            let std = [0.26862954, 0.26130258, 0.27577711][c];
                            let v = (resized[(x, y)][c] as f32 / 255.0 - mean) / std;
                            data[idx] = v;*/
                            data[idx] = imgs_arr[img_vec_idx]
                        }
                    }
                }
            }*/
            /*println!("image u8: \n");
            for x in 0..10 {
                print!("{:?}    ", img_vec_u8[x]);
            }
            println!();*/
            /*println!("after loop: \n");
            for x in 0..10 {
                print!("{:?}    ", data[x]);
            }
            println!();*/
            
            /*println!(
                "data length: {:#?}\ndata: {:?}\ntensor length: {:#?}",
                data.len(),
                data.get(0..10),
                inp_raw.nelements()
            );*/

            // Convert the f32 data into a slice of bytes.
            /*let src_slice: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<f32>(),
                )
            };
            unsafe { inp_raw.write_data(&src_slice) }*/

            unsafe { inp_raw.write_data(bytemuck::cast_slice(&data)) }

            /*let mut dst_slice = vec![0u8; batch_size * 3 * 224 * 224 * std::mem::size_of::<f32>()];
            unsafe { inp_raw.read_data(0, &mut dst_slice) }

            // Now, convert the bytes in 'dst_slice' back to the original f32 values.
            let num_elements = dst_slice.len() / std::mem::size_of::<f32>();
            let mut result = vec![0.0f32; num_elements];

            // Safety: We are transmuting the bytes in dst_slice to f32 values. Be careful to ensure the correct size and alignment.
            unsafe {
                let dst_ptr = result.as_mut_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(dst_slice.as_ptr(), dst_ptr, dst_slice.len());
            }

            println!("result: {:?}", result.get(0..10));*/

            //println!("inp raw: {:?}", get_f32_data_from_tensor(&inp_raw).get(0..20));

            //assert_eq!(self.vision_patch_embeddings.get_ne()[0], 1024);
            //assert_eq!(self.vision_patch_embeddings.get_ne()[1], 2304);
            assert_eq!(self.vision_patch_embeddings.get_ne()[0] as usize, v_patch_size * v_patch_size);
            //assert_eq!(self.vision_patch_embeddings.get_ne()[1], 2304);
            let patch_embeddings = ctx0.op_reshape_4d(
                &self.vision_patch_embeddings,
                //768, 3, 32, 32
                //16, 16, 3, 768,
                //32, 32, 3, 768,
                v_patch_size, v_patch_size, 3, self.vision_patch_embeddings.get_ne()[1] as usize / 3,
                //14, 14, 3, 1024
            );
            assert_eq!(patch_embeddings.get_ne()[2], inp_raw.get_ne()[2]);
            //let z = get_f32_data_from_tensor(&patch_embeddings);
            //println!("Debug patch embeddings 2d: {:?}", get_f32_data_from_tensor(&self.vision_patch_embeddings).get(0..20));
            //println!("Debug 0 inp: {:?}", z.get(z.len()-50..z.len()));
            //println!("Debug patch embeddings 4d: {:?}", get_f32_data_from_tensor(&patch_embeddings).get(0..20));
            //assert_eq!(self.vision_position_embeddings.nbytes(), 50 * 768 * 4);
            //assert_eq!(self.vision_position_embeddings.nelements(), 50 * 768);
            //assert_eq!(self.vision_position_embeddings.nelements(), 197 * 768);
            //let o = get_f32_data_from_tensor(&self.vision_position_embeddings);
            //println!("Debug position embeddings: {:?}", o.get(0..20));
            //assert_eq!(patch_embeddings.get_type(), ggml::Type::Q4_0);
            //assert_eq!(v_patch_size, 16);
            assert_eq!(patch_embeddings.get_type(), ggml::Type::F16);
            assert_eq!(inp_raw.get_type(), ggml::Type::F32);
            let mut inp = ctx0.op_conv_2d(
                &patch_embeddings,
                &inp_raw,
                v_patch_size, v_patch_size,
                0, 0,
                1, 1,
            );

            //assert_eq!(inp.get_ne()[0], 7);
            //assert_eq!(inp.get_ne()[1], 7);
            //assert_eq!(inp.get_ne()[2], 768);
            assert_eq!(inp.get_ne()[3] as usize, IMAGE_BATCH_SIZE);


            assert_eq!(inp.get_nb()[0], 4);
            //assert_eq!(inp.get_nb()[1], 56);
            //assert_eq!(inp.get_nb()[2], 784);
            //assert_eq!(inp.get_nb()[3], 602112);

            //println!("Debug 1 inp: {:?}", get_f32_data_from_tensor(&inp).get(0..20));

            inp = ctx0.op_reshape_3d(&inp, num_patches, v_n_embd, batch_size);
            inp = ctx0.op_cont(&ctx0.op_permute(&inp, (1, 0, 2, 3)));
            //println!("Debug 2 inp: {:?}", get_f32_data_from_tensor(&inp).get(0..20));

            // concat class_embeddings and patch_embeddings
            let mut embeddings = ctx0.new_tensor_3d(ggml::Type::F32, v_n_embd, num_positions, batch_size);

            embeddings.zero_data();
            let temp = ctx0.new_tensor_3d(ggml::Type::F32, v_n_embd, 1, batch_size);

            embeddings = ctx0.op_acc(
                &embeddings,
                &ctx0.op_repeat(&self.vision_class_embedding, &temp),
                embeddings.get_nb()[1],
                embeddings.get_nb()[2],
                embeddings.get_nb()[3],
                0,
            );
            embeddings = ctx0.op_acc(
                &embeddings,
                &inp,
                embeddings.get_nb()[1],
                embeddings.get_nb()[2],
                embeddings.get_nb()[3],
                self.vision_class_embedding.get_nb()[1],
            );
            //println!("Debug 3 embeddings: {:?}", get_f32_data_from_tensor(&embeddings).get(0..20));

            /*let positions = ctx0.new_tensor_1d(ggml::Type::I32, num_positions);
            for i in 0..num_positions {
                ggml::set_i32_1d(&positions, i, i);
            }*/
            let mut position_buf: Vec<i32> = (0..num_positions).map(|i| i as i32).collect();
            let mut positions = ctx0.new_tensor_1d(ggml::Type::I32, num_positions);
            unsafe { positions.write_data(bytemuck::cast_slice(&position_buf)) }

            assert_eq!(positions.get_type(), ggml::Type::I32);
            embeddings = ctx0.op_add(
                &embeddings,
                &ctx0.op_repeat(
                    &ctx0.op_get_rows(&self.vision_position_embeddings, &positions),
                    &embeddings,
                ),
            );

            // pre-layernorm
            {
                embeddings = ctx0.op_norm(&embeddings);

                embeddings = ctx0.op_add(
                    &ctx0.op_mul(&embeddings, &self.pre_ln_w),
                    &self.pre_ln_b,
                );
            }

            //std::process::exit(0);
            //let mut current: Tensor;
            for il in 0..v_n_layer {
                ctx0.set_offloading(self.params.should_offload(il));

                let mut current = embeddings.share();
                //let nb_q_w = self.layers[il].q_w.get_nb()[0];
                // attention uses first scratch buffer
                ctx0.use_scratch(builder.get_scratch(0));

                // layernorm1
                {
                    current = ctx0.op_norm(&current);

                    current = ctx0.op_add(
                        &ctx0.op_mul(
                            &current,
                            &self.layers[il].ln_1_w
                        ),
                        &self.layers[il].ln_1_b,
                    );
                }

                // self-attention
                {
                    let mut Q = ctx0.op_add(
                        &ctx0.op_mul_mat(&self.layers[il].q_w, &current),
                        &self.layers[il].q_b
                    );

                    Q = ctx0
                        .op_scale_inplace(&Q, &ctx0.new_f32(1.0f32 / f32::sqrt(v_head_dim as f32)));
                    Q = ctx0.op_reshape_4d(&Q, v_head_dim, v_n_head, num_positions, batch_size);
                    Q = ctx0.op_cont(&ctx0.op_permute(&Q, (0, 2, 1, 3)));
                    Q = ctx0.op_reshape_3d(&Q, v_head_dim, num_positions, v_n_head * batch_size);

                    //std::process::exit(0);

                    let mut K = ctx0.op_add(
                        &ctx0.op_mul_mat(&self.layers[il].k_w, &current),
                        &self.layers[il].k_b
                    );

                    K = ctx0.op_reshape_4d(&K, v_head_dim, v_n_head, num_positions, batch_size);
                    K = ctx0.op_cont(&ctx0.op_permute(&K, (0, 2, 1, 3)));
                    K = ctx0.op_reshape_3d(&K, v_head_dim, num_positions, v_n_head * batch_size);

                    let mut V = ctx0.op_add(
                        &ctx0.op_mul_mat(&self.layers[il].v_w, &current),
                        &self.layers[il].v_b
                    );

                    V = ctx0.op_reshape_4d(&V, v_head_dim, v_n_head, num_positions, batch_size);
                    V = ctx0.op_cont(&ctx0.op_permute(&V, (1, 2, 0, 3)));
                    V = ctx0.op_reshape_3d(&V, num_positions, v_head_dim, v_n_head * batch_size);

                    let mut KQ = ctx0.op_mul_mat(&K, &Q);
                    KQ = ctx0.op_soft_max_inplace(&KQ);
                    println!("V shape: {:?}", V.get_ne());
                    println!("KQ shape: {:?}", KQ.get_ne());
                    let mut KQV = ctx0.op_mul_mat(&V, &KQ);
                    KQV = ctx0.op_reshape_4d(&KQV, v_head_dim, num_positions, v_n_head, batch_size);
                    KQV = ctx0.op_cont(&ctx0.op_permute(&KQV, (0, 2, 1, 3)));

                    current = ctx0.op_cpy(
                        &KQV,
                        &ctx0.new_tensor_3d(ggml::Type::F32, v_n_embd, num_positions, batch_size),
                    );
                }

                // attention output
                current = ctx0.op_add(
                    &ctx0.op_mul_mat(&self.layers[il].o_w, &current),
                    &self.layers[il].o_b
                );

                // use the second scratch for the feed forward
                ctx0.use_scratch(builder.get_scratch(1));

                // re-add the layer input, e.g., residual
                current = ctx0.op_add(&current, &embeddings);

                embeddings = current.share(); // embeddings = residual, cur = hidden_states

                // layernorm2
                {
                    current = ctx0.op_norm(&current);

                    current = ctx0.op_add(
                        &ctx0.op_mul(&current, &self.layers[il].ln_2_w),
                        &self.layers[il].ln_2_b,
                    );
                }

                current = ctx0.op_mul_mat(&self.layers[il].ff_i_w, &current);
                current = ctx0.op_add(&current, &self.layers[il].ff_i_b);

                //assert_eq!(use_gelu, false);
                if use_gelu {
                    //current = ctx0.op_gelu_inplace(&current);
                    current = ctx0.op_gelu_inplace(&current);
                } else {
                    current = ctx0.op_gelu_quick_inplace(&current);
                    //current = ctx0.op_gelu_quick(&current);
                }

                current = ctx0.op_mul_mat(&self.layers[il].ff_o_w, &current);
                current = ctx0.op_add(&current, &self.layers[il].ff_o_b);

                // residual 2
                current = ctx0.op_add(&embeddings, &current);

                embeddings = current;

                //std::process::exit(0);
            }

            // get the output of cls token, e.g., 0th index
            let mut cls_buf: Vec<i32> = (0..batch_size).map(|b| (b * num_positions) as i32).collect();
            let mut cls = ctx0.new_tensor_1d(ggml::Type::I32, batch_size);
            unsafe { cls.write_data(bytemuck::cast_slice(&cls_buf)) }
            /*
            struct ggml_tensor *cls = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);
            for (int b = 0; b < batch_size; b++)
            {
                ggml_set_i32_1d(cls, b, b * num_positions);
            }
            embeddings = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, embeddings, hidden_size, num_positions * batch_size), cls);
            let cls = ctx0.new_tensor_1d(ggml::Type::I32, batch_size);
            for b in 0..batch_size {
                ggml::set_i32_1d(&cls, b, b * num_positions);
            }*/
            embeddings = ctx0.op_get_rows(
                &ctx0.op_reshape_2d(&embeddings, v_n_embd, num_positions * batch_size),
                &cls,
            );

            // use the first scratch for the norm
            ctx0.use_scratch(builder.get_scratch(0));

            // post-layernorm
            {
                embeddings = ctx0.op_norm(&embeddings);

                embeddings = ctx0.op_add(
                    &ctx0.op_mul(&embeddings, &self.post_ln_w),
                    &self.post_ln_b,
                );
            }
            
            ctx0.set_offloading(false);

            //ctx0.use_scratch(builder.get_scratch(0));
            ctx0.use_scratch(None);

            // final visual projection
            embeddings = ctx0.op_mul_mat(&self.projection, &embeddings);

            //println!("embeddings: {:?}", get_f32_data_from_tensor(&embeddings).get(0..20));

            // normalize output embeddings
            let mut output = ctx0.new_tensor_2d(ggml::Type::F32, v_projection_dim, batch_size);
            
            for b in 0..batch_size {
                let mut embedding = ctx0.op_get_rows(&embeddings, &ctx0.new_i32(b as i32));
                let length = ctx0.op_sqrt(&ctx0.op_sum(&ctx0.op_sqr(&embedding)));
                embedding = ctx0.op_scale_inplace(&embedding, &ctx0.op_div(&ctx0.new_f32(1.0f32), &length));
                output = ctx0.op_acc(
                    &output,
                    &embedding,
                    output.get_nb()[1],
                    output.get_nb()[2],
                    output.get_nb()[3],
                    b * embedding.nbytes(),
                );

                //let image_embeddings = get_f32_data_from_tensor(&output);
                //println!("image_embeddings(batch: {:?}): {:?}", b, image_embeddings.get(0..10));
            }
            //ggml_set_name(output, "check");
            assert_eq!(output.get_ne()[0] as usize, v_projection_dim);
            assert_eq!(output.get_ne()[1] as usize, IMAGE_BATCH_SIZE);
            //xxctx0.set_offloading(false);
            //assert_eq!(output.nbytes(), 2048 * IMAGE_BATCH_SIZE);
            //assert_eq!(output.nelements(), 512 * IMAGE_BATCH_SIZE);
            // read output
            //let mut dst_slice = vec![0u8; batch_size * 3 * 224 * 224 * std::mem::size_of::<f32>()];
            /*fn get_f32_data_from_tensor(a: &Tensor) -> Vec<f32> {
                let mut dst_slice = vec![0u8; a.nbytes()];
                unsafe { a.read_data(0, &mut dst_slice) }

                // Now, convert the bytes in 'dst_slice' back to the original f32 values.
                //let num_elements = dst_slice.len() / std::mem::size_of::<f32>();
                //let num_elements = output.nelements();
                //let mut data_vec = { vec![9f32; a.nelements()] } if a.get_type() == ggml::Type::F32 else { vec![9f16; a.nelements()] };
                let mut data_vec = vec![9f32; a.nelements()];

                // Safety: We are transmuting the bytes in dst_slice to f32 values. Be careful to ensure the correct size and alignment.
                unsafe {
                    let dst_ptr = data_vec.as_mut_ptr() as *mut u8;
                    std::ptr::copy_nonoverlapping(dst_slice.as_ptr(), dst_ptr, dst_slice.len());
                }
                data_vec
            }*/

            fn get_f32_data_from_tensor(a: &Tensor) -> Vec<f32> {
                let mut data = vec![9.0; a.nbytes()];

                unsafe {
                    a.read_data(0, bytemuck::cast_slice_mut(&mut data));
                }
                data
            }

            //let image_embeddings = get_f32_data_from_tensor(&output);
            //println!("image_embeddings: {:?}", image_embeddings.get(0..20));
            //std::process::exit(0);

            /*if image_embeddings[0] > 0.0 {
                std::process::exit(0);
            } else {
                println!("embeddings is 0")
            }*/

            (
                gf,
                GraphOutputs {
                    result: output.share(),
                    embedding_result: output.share(),
                    //embedding_result: embeddings_tensor,
                },
            )
        });

        // finish evaluation
        //common::read_last_token(session, &outputs.result, t_n_vocab, IMAGE_BATCH_SIZE);
        //common::extract_logits(output_request, &outputs.result, t_n_vocab, IMAGE_BATCH_SIZE);
        let v_n_embd = v_projection_dim;
        extract_embeddings(output_request, &outputs.embedding_result, v_n_embd, IMAGE_BATCH_SIZE);
        //std::process::exit(0);
        /// Extract embeddings from [OutputRequest] evaluation
        pub fn extract_embeddings(
            output_request: &mut OutputRequest,
            embeddings_tensor: &Tensor,
            n_embd: usize,
            n: usize,
        ) {
            // Extract embeddings
            if let Some(embeddings) = &mut output_request.embeddings {
                //assert_eq!(n_embd, v_projection_dim);
                //assert_eq!(n, IMAGE_BATCH_SIZE);
                embeddings.resize(n_embd * n, 9.0);
                // Create a new vector to hold all embeddings
                let mut all_embeddings = vec![9.0; n_embd * n];
                // SAFETY: Same rationale as for the "Extract logits" section applies.
                assert_eq!(embeddings_tensor.nelements(), n_embd * n);
                unsafe {
                    embeddings_tensor.read_data(0, bytemuck::cast_slice_mut(&mut all_embeddings));
                }
                embeddings.copy_from_slice(&all_embeddings[n_embd * 0..n_embd * n]);
                //let imgs: Vec<Vec<u32>> = 
                /*let embeddings = embeddings.chunks(512)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<Vec<f32>>>();*/
            }
        }
    }

    #[tracing::instrument(level = "trace", skip_all)]
    fn batch_evaluate(
        &self,
        session: &mut InferenceSession,
        input_batch_tokens: &[&[u32]],
        output_request: &mut OutputRequest,
    ) -> Result<Vec<Vec<f32>>, String> {
        let input_tokens = input_batch_tokens.concat();
        self.evaluate(session, &input_tokens, output_request);
        let hyperparameters = self.hyperparameters();
        let embedding_dim = hyperparameters.v_projection_dim;
        let batch_embeddings = output_request.embeddings.as_ref().unwrap()
            .chunks(embedding_dim)
            .map(|chunk| chunk.to_vec())
            .collect();
        Ok(batch_embeddings)
    }

    fn hyperparameters(&self) -> &Self::Hyperparameters {
        &self.hyperparameters
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn context_size(&self) -> usize {
        self.params.context_size
    }

    fn bot_token_id(&self) -> Option<TokenId> {
        None
    }

    fn eot_token_id(&self) -> TokenId {
        self.tokenizer.id("<|endoftext|>".as_bytes()).unwrap()
    }

    fn quantize_tensors() -> Vec<Regex> {
        vec![Regex::new(".*weight").unwrap()]
    }

    fn skip_quantize_tensors() -> Vec<Regex> {
        //vec![]
        //vec![Regex::new(".*vision_model.embeddings.patch_embedding.weight").unwrap()]
        vec![Regex::new(".*patch_embedding.weight").unwrap()]
    }

    fn supports_rewind(&self) -> bool {
        true
    }
}

/// GPT-NeoX [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Hyperparameters {
    /// text
    /// Size of the model's vocabulary
    pub t_n_vocab: usize,
    /// Size of the model's context
    pub t_n_pos: usize,
    /// Size of the model's embedding layer
    pub t_n_embd: usize,
    ///
    pub t_n_intermediate: usize,
    ///
    pub t_projection_dim: usize,
    /// n_head
    pub t_n_head: usize,
    /// Number of layers in the model
    pub t_n_layer: usize,

    /// vision
    /// Size of the model's vocabulary
    pub v_image_size: usize,
    /// Size of the model's context
    pub v_patch_size: usize,
    /// Size of the model's embedding layer
    pub v_n_embd: usize,
    ///
    pub v_n_intermediate: usize,
    ///
    pub v_projection_dim: usize,
    /// n_head
    pub v_n_head: usize,
    /// Number of layers in the model
    pub v_n_layer: usize,

    /// others
    pub use_gelu: bool,
    /// file_type
    pub file_type: FileType,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            // text
            t_n_vocab: Default::default(),
            t_n_pos: Default::default(),
            t_n_embd: Default::default(),
            t_n_intermediate: Default::default(),
            t_projection_dim: Default::default(),
            t_n_head: Default::default(),
            t_n_layer: Default::default(),
            // vision
            v_image_size: Default::default(),
            v_patch_size: Default::default(),
            v_n_embd: Default::default(),
            v_n_intermediate: Default::default(),
            v_projection_dim: Default::default(),
            v_n_head: Default::default(),
            v_n_layer: Default::default(),

            file_type: Default::default(),
            use_gelu: true,
        }
    }
}

impl llm_base::Hyperparameters for Hyperparameters {
    fn read_ggml(reader: &mut dyn std::io::BufRead) -> Result<Self, LoadError> {
        Ok(Hyperparameters {
            // text params
            t_n_vocab: util::read_i32(reader)?.try_into()?,
            t_n_pos: util::read_i32(reader)?.try_into()?,
            t_n_embd: util::read_i32(reader)?.try_into()?,
            t_n_intermediate: util::read_i32(reader)?.try_into()?,
            t_projection_dim: util::read_i32(reader)?.try_into()?,
            t_n_head: util::read_i32(reader)?.try_into()?,
            t_n_layer: util::read_i32(reader)?.try_into()?,
            // vision params
            v_image_size: util::read_i32(reader)?.try_into()?,
            v_patch_size: util::read_i32(reader)?.try_into()?,
            v_n_embd: util::read_i32(reader)?.try_into()?,
            v_n_intermediate: util::read_i32(reader)?.try_into()?,
            v_projection_dim: util::read_i32(reader)?.try_into()?,
            v_n_head: util::read_i32(reader)?.try_into()?,
            v_n_layer: util::read_i32(reader)?.try_into()?,
            // others
            use_gelu: util::read_bool(reader)?,
            file_type: util::read_filetype(reader)?,
        })
    }

    fn write_ggml(&self, writer: &mut dyn std::io::Write) -> Result<(), HyperparametersWriteError> {
        // text
        util::write_i32(writer, self.t_n_vocab.try_into()?)?;
        util::write_i32(writer, self.t_n_pos.try_into()?)?;
        util::write_i32(writer, self.t_n_embd.try_into()?)?;
        util::write_i32(writer, self.t_n_intermediate.try_into()?)?;
        util::write_i32(writer, self.t_projection_dim.try_into()?)?;
        util::write_i32(writer, self.t_n_head.try_into()?)?;
        util::write_i32(writer, self.t_n_layer.try_into()?)?;
        // vision
        util::write_i32(writer, self.v_image_size.try_into()?)?;
        util::write_i32(writer, self.v_patch_size.try_into()?)?;
        util::write_i32(writer, self.v_n_embd.try_into()?)?;
        util::write_i32(writer, self.v_n_intermediate.try_into()?)?;
        util::write_i32(writer, self.v_projection_dim.try_into()?)?;
        util::write_i32(writer, self.v_n_head.try_into()?)?;
        util::write_i32(writer, self.v_n_layer.try_into()?)?;
        // others
        util::write_bool(writer, self.use_gelu)?;
        util::write_i32(writer, self.file_type.into())?;
        Ok(())
    }

    fn n_vocabulary(&self) -> usize {
        self.t_n_vocab
    }

    fn file_type(&self) -> Option<FileType> {
        Some(self.file_type)
    }

    fn file_type_mut(&mut self) -> Option<&mut FileType> {
        Some(&mut self.file_type)
    }
}

struct Layer {
    // pre-normalization
    ln_1_w: Tensor,
    ln_1_b: Tensor,

    // attention
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    o_w: Tensor,
    o_b: Tensor,

    ff_i_w: Tensor,
    ff_i_b: Tensor,
    ff_o_w: Tensor,
    ff_o_b: Tensor,

    // post normalization
    ln_2_w: Tensor,
    ln_2_b: Tensor,
}
