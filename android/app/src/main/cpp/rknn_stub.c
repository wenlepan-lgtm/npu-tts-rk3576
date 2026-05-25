/*
 * RKNN API stub functions
 *
 * Provides weak symbol definitions so the project compiles on non-RKNN devices
 * (standard Android emulators, x86 test builds, etc.)
 * When running on actual RK3576 hardware, the real librknnrt.so takes precedence.
 */

typedef void* rknn_context;

// Weak stubs — overridden by real librknnrt.so if linked
__attribute__((weak)) int rknn_init(rknn_context* ctx, const char* model_path,
                                     void* mem, size_t size, int flag) {
    (void)ctx; (void)model_path; (void)mem; (void)size; (void)flag;
    return -1;  // Not available
}

__attribute__((weak)) int rknn_destroy(rknn_context ctx) {
    (void)ctx;
    return -1;
}

__attribute__((weak)) int rknn_inputs_set(rknn_context ctx, int n_inputs,
                                           void* inputs) {
    (void)ctx; (void)n_inputs; (void)inputs;
    return -1;
}

__attribute__((weak)) int rknn_run(rknn_context ctx, void* rknn_input,
                                    void* rknn_output) {
    (void)ctx; (void)rknn_input; (void)rknn_output;
    return -1;
}

__attribute__((weak)) int rknn_outputs_get(rknn_context ctx, int n_outputs,
                                            void* outputs, void* pass_through) {
    (void)ctx; (void)n_outputs; (void)outputs; (void)pass_through;
    return -1;
}

__attribute__((weak)) int rknn_outputs_release(rknn_context ctx, int n_outputs,
                                                void* outputs) {
    (void)ctx; (void)n_outputs; (void)outputs;
    return -1;
}

// RKNN tensor type constants (matching rknn_api.h)
#define RKNN_TENSOR_FLOAT32 0
#define RKNN_TENSOR_NHWC    0

// Stub struct definitions
typedef struct {
    int index;
    int type;
    int fmt;
    size_t size;
    void* buf;
} rknn_input_stub;
typedef rknn_input_stub rknn_input;

typedef struct {
    int index;
    int want_float;
    size_t size;
    void* buf;
} rknn_output_stub;
typedef rknn_output_stub rknn_output;
