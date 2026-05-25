#include "onnx_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <onnxruntime_cxx_api.h>

struct OnnxEncoder {
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions session_opts;
    std::vector<float> g_embedding;  // speaker embedding from g.bin
    int initialized;

    OnnxEncoder()
        : env(ORT_LOGGING_LEVEL_WARNING, "npu-tts"),
          session(nullptr),
          initialized(0) {}
};

static std::vector<float> load_g_bin(const char* g_path) {
    std::vector<float> embedding;
    if (!g_path) return embedding;

    FILE* f = fopen(g_path, "rb");
    if (!f) {
        fprintf(stderr, "onnx_encoder: cannot open g.bin: %s\n", g_path);
        return embedding;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    int count = size / sizeof(float);
    embedding.resize(count);
    fread(embedding.data(), sizeof(float), count, f);
    fclose(f);

    return embedding;
}

OnnxEncoder* encoder_init(const char* model_path, const char* g_path) {
    if (!model_path) return nullptr;

    OnnxEncoder* enc = new OnnxEncoder();

    try {
        enc->session_opts.SetIntraOpNumThreads(2);
        enc->session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        const char* model_paths[] = { model_path };
        enc->session = Ort::Session(enc->env, model_paths[0], enc->session_opts);
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "onnx_encoder: failed to create session: %s\n", e.what());
        delete enc;
        return nullptr;
    }

    enc->g_embedding = load_g_bin(g_path);
    enc->initialized = 1;
    return enc;
}

int encoder_run(OnnxEncoder* enc,
                const int* phones, const int* tones, const int* langs,
                int seq_len,
                float noise_scale, float length_scale,
                struct EncoderOutput* output) {
    if (!enc || !enc->initialized || !phones || !tones || !langs || !output) {
        return -1;
    }

    memset(output, 0, sizeof(*output));

    try {
        Ort::AllocatorWithDefaultOptions alloc;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Prepare input tensors
        // MeloTTS encoder inputs: phones(1,T), tones(1,T), langs(1,T), noise_scale, length_scale, g(1,256)
        std::vector<int64_t> seq_shape = {1, seq_len};

        auto phones_tensor = Ort::Value::CreateTensor<int32_t>(
            memory_info, const_cast<int32_t*>(reinterpret_cast<const int32_t*>(phones)),
            seq_len, seq_shape.data(), seq_shape.size());

        auto tones_tensor = Ort::Value::CreateTensor<int32_t>(
            memory_info, const_cast<int32_t*>(reinterpret_cast<const int32_t*>(tones)),
            seq_len, seq_shape.data(), seq_shape.size());

        auto langs_tensor = Ort::Value::CreateTensor<int32_t>(
            memory_info, const_cast<int32_t*>(reinterpret_cast<const int32_t*>(langs)),
            seq_len, seq_shape.data(), seq_shape.size());

        int64_t scalar_shape[] = {};
        auto noise_tensor = Ort::Value::CreateTensor<float>(
            memory_info, &noise_scale, 1, scalar_shape, 0);

        auto length_tensor = Ort::Value::CreateTensor<float>(
            memory_info, &length_scale, 1, scalar_shape, 0);

        // Speaker embedding g
        std::vector<Ort::Value> inputs;
        std::vector<const char*> input_names;
        input_names.push_back("phones");
        inputs.push_back(std::move(phones_tensor));
        input_names.push_back("tones");
        inputs.push_back(std::move(tones_tensor));
        input_names.push_back("langs");
        inputs.push_back(std::move(langs_tensor));
        input_names.push_back("noise_scale");
        inputs.push_back(std::move(noise_tensor));
        input_names.push_back("length_scale");
        inputs.push_back(std::move(length_tensor));

        if (!enc->g_embedding.empty()) {
            int64_t g_shape[] = {1, (int64_t)enc->g_embedding.size()};
            auto g_tensor = Ort::Value::CreateTensor<float>(
                memory_info, enc->g_embedding.data(),
                enc->g_embedding.size(), g_shape, 2);
            input_names.push_back("g");
            inputs.push_back(std::move(g_tensor));
        }

        // Run encoder
        const char* output_names[] = {"z", "y_mask"};
        auto results = enc->session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(), inputs.data(), input_names.size(),
            output_names, 2);

        // Extract z
        auto& z_result = results[0];
        auto z_info = z_result.GetTensorTypeAndShapeInfo();
        auto z_shape = z_info.GetShape();
        size_t z_count = z_info.GetElementCount();
        const float* z_data = z_result.GetTensorData<float>();

        output->z = (float*)malloc(z_count * sizeof(float));
        memcpy(output->z, z_data, z_count * sizeof(float));
        output->z_len = (int)z_count;

        // Extract y_mask
        auto& mask_result = results[1];
        auto mask_info = mask_result.GetTensorTypeAndShapeInfo();
        size_t mask_count = mask_info.GetElementCount();
        const float* mask_data = mask_result.GetTensorData<float>();

        output->y_mask = (float*)malloc(mask_count * sizeof(float));
        memcpy(output->y_mask, mask_data, mask_count * sizeof(float));
        output->mask_len = (int)mask_count;

        // Sequence length from shape
        if (z_shape.size() >= 2) {
            output->T = (int)z_shape[z_shape.size() - 1];
        }

        return 0;
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "onnx_encoder: run failed: %s\n", e.what());
        return -1;
    }
}

void encoder_free_output(struct EncoderOutput* output) {
    if (!output) return;
    free(output->z);
    free(output->y_mask);
    memset(output, 0, sizeof(*output));
}

void encoder_release(OnnxEncoder* enc) {
    if (!enc) return;
    delete enc;
}
