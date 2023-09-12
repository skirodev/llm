bool clip_image_batch_encode(const clip_ctx * ctx, int n_threads, const std::vector<clip_image_f32> & imgs, float * vec) {
    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    const int image_size = hparams.image_size;
    const int patch_size = hparams.patch_size;
    const int num_patches = ((image_size / patch_size) * (image_size / patch_size));
    const int num_positions = num_patches + 1;
    const int hidden_size = hparams.hidden_size;
    const int n_head = hparams.n_head;
    const int d_head = hidden_size / n_head;
    const int n_layer = hparams.n_layer;
    const int n_intermediate = hparams.n_intermediate;
    const int projection_dim = hparams.projection_dim;
    int batch_size = imgs.size();

    auto & buf_compute = ctx->buf_compute;

    struct ggml_init_params params = {
        .mem_size = buf_compute.size,
        .mem_buffer = buf_compute.data,
        .no_alloc = false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};

    static size_t scr0_size =
        get_scr_buf_req_by_size(ctx->text_model.tensors.size() + ctx->vision_model.tensors.size(), num_positions);
    static void * scr0 = malloc(scr0_size);

    struct ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size, image_size, 3, batch_size);

    {
        float * data = (float *)ggml_get_data(inp_raw);

        for (int b = 0; b < imgs.size(); b++) {
            const int nx = imgs[b].nx;
            const int ny = imgs[b].ny;
            GGML_ASSERT(nx == image_size && ny == image_size);

            const int n = nx * ny;

            for (int b = 0; b < batch_size; b++) {
                for (int k = 0; k < 3; k++) {
                    for (int y = 0; y < ny; y++) {
                        for (int x = 0; x < nx; x++) {
                            data[(b * 3 * n) + k * n + y * nx + x] = imgs[b].data[3 * (y * nx + x) + k];
                        }
                    }
                }
            }
        }
    }

    struct ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    inp = ggml_reshape_3d(ctx0, inp, num_patches, hidden_size, batch_size);
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

    // concat class_embeddings and patch_embeddings
    struct ggml_tensor * embeddings = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, num_positions, batch_size);

    ggml_set_zero(embeddings);
    struct ggml_tensor * temp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, 1, batch_size);

    embeddings = ggml_acc(ctx0, embeddings, ggml_repeat(ctx0, model.class_embedding, temp), embeddings->nb[1],
                          embeddings->nb[2], embeddings->nb[3], 0);
    embeddings =
        ggml_acc(ctx0, embeddings, inp, embeddings->nb[1], embeddings->nb[2], embeddings->nb[3], model.class_embedding->nb[1]);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_positions);
    for (int i = 0; i < num_positions; i++) {
        ggml_set_i32_1d(positions, i, i);
    }

    embeddings =
        ggml_add(ctx0, embeddings, ggml_repeat(ctx0, ggml_get_rows(ctx0, model.position_embeddings, positions), embeddings));

    // pre-layernorm
    {
        embeddings = ggml_norm(ctx0, embeddings);

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.pre_ln_w, embeddings), embeddings),
                              ggml_repeat(ctx0, model.pre_ln_b, embeddings));
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        struct ggml_tensor * cur = embeddings; // embeddings = residual, cur = hidden_states

        const size_t nb_q_w = model.layers[il].q_w->nb[0];

        ggml_set_scratch(ctx0, {0, scr0_size, scr0});

        // layernorm1
        {
            cur = ggml_norm(ctx0, cur);

            cur = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[il].ln_1_w, cur), cur),
                           ggml_repeat(ctx0, model.layers[il].ln_1_b, cur));
        }

        // self-attention
        {

            struct ggml_tensor * Q =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].q_b, cur), ggml_mul_mat(ctx0, model.layers[il].q_w, cur));

            Q = ggml_scale_inplace(ctx0, Q, ggml_new_f32(ctx0, 1.0f / sqrt((float)d_head)));
            Q = ggml_reshape_4d(ctx0, Q, d_head, n_head, num_positions, batch_size);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            Q = ggml_reshape_3d(ctx0, Q, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * K =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].k_b, cur), ggml_mul_mat(ctx0, model.layers[il].k_w, cur));

            K = ggml_reshape_4d(ctx0, K, d_head, n_head, num_positions, batch_size);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            K = ggml_reshape_3d(ctx0, K, d_head, num_positions, n_head * batch_size);

            struct ggml_tensor * V =
                ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].v_b, cur), ggml_mul_mat(ctx0, model.layers[il].v_w, cur));

            V = ggml_reshape_4d(ctx0, V, d_head, n_head, num_positions, batch_size);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
            V = ggml_reshape_3d(ctx0, V, num_positions, d_head, n_head * batch_size);

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_soft_max_inplace(ctx0, KQ);
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_reshape_4d(ctx0, KQV, d_head, num_positions, n_head, batch_size);
            KQV = ggml_cont(ctx0, ggml_permute(ctx0, KQV, 0, 2, 1, 3));

            cur = ggml_cpy(ctx0, KQV, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hidden_size, num_positions, batch_size));
        }

        // attention output
        cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].o_b, cur), ggml_mul_mat(ctx0, model.layers[il].o_w, cur));

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, embeddings);

        embeddings = cur; // embeddings = residual, cur = hidden_states

        // layernorm2
        {
            cur = ggml_norm(ctx0, cur);

            cur = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[il].ln_2_w, cur), cur),
                           ggml_repeat(ctx0, model.layers[il].ln_2_b, cur));
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
        cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].ff_i_b, cur), cur);

        if (ctx->use_gelu) {
            cur = ggml_gelu_inplace(ctx0, cur);
        } else {
            cur = ggml_gelu_quick_inplace(ctx0, cur);
        }

        cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
        cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].ff_o_b, cur), cur);

        // residual 2
        cur = ggml_add(ctx0, embeddings, cur);

        embeddings = cur;
    }

    // get the output of cls token, e.g., 0th index
    struct ggml_tensor * cls = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, batch_size);
    for (int b = 0; b < batch_size; b++) {
        ggml_set_i32_1d(cls, b, b * num_positions);
    }
    embeddings = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, embeddings, hidden_size, num_positions * batch_size), cls);

    // post-layernorm
    {
        embeddings = ggml_norm(ctx0, embeddings);

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.post_ln_w, embeddings), embeddings),
                              ggml_repeat(ctx0, model.post_ln_b, embeddings));
    }

    ggml_set_scratch(ctx0, {0, 0, nullptr});

    // final visual projection
    embeddings = ggml_mul_mat(ctx0, model.projection, embeddings);

    // normalize output embeddings
    struct ggml_tensor * output = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, projection_dim, batch_size);

    for (int b = 0; b < batch_size; b++) {
        struct ggml_tensor * embedding = ggml_get_rows(ctx0, embeddings, ggml_new_i32(ctx0, b));
        ggml_tensor * length = ggml_sqrt(ctx0, ggml_sum(ctx0, ggml_sqr(ctx0, embedding)));
        embedding = ggml_scale_inplace(ctx0, embedding, ggml_div(ctx0, ggml_new_f32(ctx0, 1.0f), length));
        output = ggml_acc(ctx0, output, embedding, output->nb[1], output->nb[2], output->nb[3], b * ggml_nbytes(embedding));
    }
    ggml_set_name(output, "check");

    // run the computation
    ggml_build_forward_expand(&gf, output);
    ggml_cplan cplan = ggml_graph_plan(&gf, n_threads);
    if (cplan.work_size != 0) {
        cplan.work_data = (uint8_t *)malloc(cplan.work_size);
    }
    ggml_graph_compute(&gf, &cplan);

// print
#ifdef CLIP_DEBUG
    {
        auto print_t_f32 = [&](struct ggml_tensor * t) {
            float * data = (float *)t->data;
            printf("dtype: f32, dims: %jd %jd %jd %jd, nb: %jd %jd %jd %jd\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[0],
                   t->nb[1], t->nb[2], t->nb[3]);
            printf("data: ");
            for (int i = 0; i < std::min((int)t->ne[0], 20); i++) {
                printf("%f ", data[i]);
            }

            // printf("\n\n");
            double sum = 0.0;
            for (int i = 0; i < ggml_nelements(t); i++) {
                sum += data[i];
            }
            printf("sum:  %f\n", sum);
        };

        auto print_t_f16 = [&](struct ggml_tensor * t) {
            ggml_fp16_t * data = (ggml_fp16_t *)t->data;
            printf("dtype: f16, dims: %jd %jd %jd %jd\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            printf("data: ");
            for (int i = 0; i < std::min((int)t->ne[0], 10); i++) {
                printf("%f ", ggml_fp16_to_fp32(data[i]));
            }
            printf("\n\n");
            double sum = 0.0;
            for (int i = 0; i < ggml_nelements(t); i++) {
                sum += ggml_fp16_to_fp32(data[i]);
            }
            printf("sum:  %f\n", sum);
        };

        auto * t = ggml_get_tensor(ctx0, "check");
        // auto t = inp_raw;
        if (t->type == GGML_TYPE_F32) {
            print_t_f32(t);
        } else {
            print_t_f16(t);
        }
    }

    printf("used_mem = %zu\n", ggml_used_mem(ctx0));
#endif

    memcpy(vec, ggml_get_data_f32(output), sizeof(float) * projection_dim * batch_size);

    if (cplan.work_size != 0) {
        free(cplan.work_data);
    }

    ggml_free(ctx0);

    return true;
}