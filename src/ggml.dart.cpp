#include "ggml.dart.h"
struct ggml_context *ggml_context_new_empty() {
  struct ggml_context *ctx = {};
  return ctx;
}
