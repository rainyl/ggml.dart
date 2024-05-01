import 'dart:ffi' as ffi;

import 'alloc.dart';
import 'struct.dart';
import 'base.dart';
import 'constants.dart';
import 'ggml.g.dart' as gg;

class GGMLBackend extends GgObject<gg.ggml_backend> {
  GGMLBackend._(super.ptr) : super.fromPtr();
  factory GGMLBackend([int type = GGML_BACKEND_TYPE_CPU]) {
    switch (type) {
      case GGML_BACKEND_TYPE_CPU:
        return GGMLBackend._(gg.ggml_backend_cpu_init());
      case GGML_BACKEND_TYPE_GPU:
        throw UnimplementedError();
      case GGML_BACKEND_TYPE_GPU_SPLIT:
        throw UnimplementedError();
      default:
        throw UnimplementedError();
    }
  }

  GGMLBackendBufferType getDefaultBufferType() {
    final p = gg.ggml_backend_get_default_buffer_type(ptr);
    return GGMLBackendBufferType.fromPtr(p);
  }

  bool isCpu() => gg.ggml_backend_is_cpu(ptr);

  void setNThreads(int nThreads) {
    if (isCpu()) {
      gg.ggml_backend_cpu_set_n_threads(ptr, nThreads);
    } else {
      throw Exception("Backend is not cpu");
    }
  }

  int graphCompute(GGMLCGraph cgraph) => gg.ggml_backend_graph_compute(ptr, cgraph.ptr);

  void tensorGet(GGMLTensor tensor, int offset, int size) {}// TODO

  static final finalizer =
      ggFinalizer<ffi.Pointer<gg.ggml_backend>>(ffi.Native.addressOf(gg.ggml_backend_free));
}
