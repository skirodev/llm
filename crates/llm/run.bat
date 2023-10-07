::cargo run --example clip-text --release -- cliptext "C:\Users\11048\Documents\cpp\chatglm.cpp\CLIP-ViT-B-32-laion2B-s34B-b79K\ggml-model-f16.bin" -r "laion/CLIP-ViT-B-32-laion2B-s34B-b79K" -q a

::cargo run --example clip-vision --release

::cargo run --example token --release

::cargo run --example clip --release
::-q "a sitting girl"
::cargo run --release -- info -a clip -m "C:\Users\11048\Documents\cpp\chatglm.cpp\CLIP-ViT-B-32-laion2B-s34B-b79K\ggml-model-f16.bin"

::cargo run --release -- quantize -a clipvision C:\Users\11048\Documents\cpp\chatglm.cpp\CLIP-ViT-B-32-laion2B-s34B-b79K\ggml-model-f16.bin C:\Users\11048\Documents\cpp\chatglm.cpp\CLIP-ViT-B-32-laion2B-s34B-b79K\ggml-model-q4_0.bin q4_0

::cargo run --release --example image
::cargo run --release --example search
::cargo run --release --example demo
::cargo run --release --example freya
::cargo run --release --example token
::set OPENCL_PATH=C:\Users\11048\Documents\vcpkg\packages\opencl_x64-windows\bin
::set CLBLAST_PATH=C:\Users\11048\Documents\vcpkg\packages\clblast_x64-windows\bin
cargo run --release --example clip-vision-batch
::cargo run --release --example vectorite