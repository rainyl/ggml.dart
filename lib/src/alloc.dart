import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'backend.dart';
import 'base.dart';
import 'struct.dart';
import 'ggml.g.dart' as gg;

class GGMLBackendBuffer extends GgObject<gg.ggml_backend_buffer> {
  GGMLBackendBuffer.fromPtr(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  static final finalizer = ggFinalizer<ffi.Pointer<gg.ggml_backend_buffer>>(ffi.Native.addressOf(gg.ggml_backend_buffer_free));
}

class GGMLBackendBufferType extends GgObject<gg.ggml_backend_buffer_type> {
  GGMLBackendBufferType.fromPtr(super.ptr) : super.fromPtr();
}

// Tensor allocator
class GGMLTallocr extends GgStruct<gg.ggml_tallocr> {
  GGMLTallocr.fromPtr(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory GGMLTallocr(GGMLBackendBuffer buffer) {
    final s = gg.ggml_tallocr_new(buffer.ptr);
    final p = calloc<gg.ggml_tallocr>()..ref = s;
    return GGMLTallocr.fromPtr(p);
  }

  void alloc(GGMLTensor tensor) => gg.ggml_tallocr_alloc(ptr, tensor.ptr);

  static final finalizer = ggFinalizer(calloc.nativeFree);

  @override
  gg.ggml_tallocr get ref => ptr.ref;
}

// Graph allocator
/*
  Example usage:
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_bacckend_cpu_buffer_type());

    // optional: create a worst-case graph and reserve the buffers to avoid reallocations
    ggml_gallocr_reserve(galloc, build_graph(max_batch));

    // allocate the graph
    struct ggml_cgraph * graph = build_graph(batch);
    ggml_gallocr_alloc_graph(galloc, graph);

    printf("compute buffer size: %zu bytes\n", ggml_gallocr_get_buffer_size(galloc, 0));

    // evaluate the graph
    ggml_backend_graph_compute(backend, graph);
*/

// special tensor flags for use with the graph allocator:
//   ggml_set_input(): all input tensors are allocated at the beginning of the graph in non-overlapping addresses
//   ggml_set_output(): output tensors are never freed and never overwritten
class GGMLGAllocr extends GgObject<gg.ggml_gallocr> {
  GGMLGAllocr.fromPtr(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory GGMLGAllocr(GGMLBackendBufferType buft, [int? nBufs]) {
    final p = nBufs == null
        ? gg.ggml_gallocr_new(buft.ptr)
        : gg.ggml_gallocr_new_n(ffi.Pointer.fromAddress(buft.ptr.address), nBufs);
    return GGMLGAllocr.fromPtr(p);
  }

  // pre-allocate buffers from a measure graph - does not allocate or modify the graph
  // call with a worst-case graph to avoid buffer reallocations
  // not strictly required for single buffer usage: ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
  // returns false if the buffer allocation failed
  bool reserve(GGMLCGraph graph) => gg.ggml_gallocr_reserve(ptr, graph.ptr);
  // bool reserveN(GgmlCGraph graph, List<int> nodeBufferIds, List<int> leafBufferIds) {
  //   return using<bool>((arena) {
  //     final nodeBufferIdsPtr = arena<ffi.Int32>(nodeBufferIds.length);
  //     final leafBufferIdsPtr = arena<ffi.Int32>(leafBufferIds.length);
  //     for (var i = 0; i < nodeBufferIds.length; i++) {
  //       nodeBufferIdsPtr[i] = nodeBufferIds[i];
  //     }
  //     for (var i = 0; i < leafBufferIds.length; i++) {
  //       leafBufferIdsPtr[i] = leafBufferIds[i];
  //     }
  //     return gg.ggml_gallocr_reserve_n(
  //         ptr, graph.ptr, nodeBufferIdsPtr, nodeBufferIds.length, leafBufferIdsPtr, leafBufferIds.length);
  //   });
  // }

  // automatic reallocation if the topology changes when using a single buffer
  // returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
  bool allocGraph(GGMLCGraph graph) => gg.ggml_gallocr_alloc_graph(ptr, graph.ptr);

  int getBufferSize(int bufId) => gg.ggml_gallocr_get_buffer_size(ptr, bufId);

  static final finalizer = ggFinalizer<ffi.Pointer<gg.ggml_gallocr>>(ffi.Native.addressOf(gg.ggml_gallocr_free));
}

// Utils
// Create a buffer and allocate all the tensors in a ggml_context
GGMLBackendBuffer allocCtxTensorsFromBuft(GGMLContext ctx, GGMLBackendBufferType buft) {
  final p = gg.ggml_backend_alloc_ctx_tensors_from_buft(ctx.ptr, buft.ptr);
  return GGMLBackendBuffer.fromPtr(p);
}

GGMLBackendBuffer allocCtxTensors(GGMLContext ctx, GGMLBackend backend) {
  final p = gg.ggml_backend_alloc_ctx_tensors(ctx.ptr, backend.ptr);
  return GGMLBackendBuffer.fromPtr(p);
}
