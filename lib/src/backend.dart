import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'constants.dart';
import 'graph.dart';
import 'tensor.dart';
import 'ggml.g.dart' as gg;

class BackendBuffer extends GGBase<gg.ggml_backend_buffer> {
  BackendBuffer.fromPtr(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  String getName() => gg.ggml_backend_buffer_name(ptr).cast<Utf8>().toDartString();
  ffi.Pointer<ffi.Void> getBase() => gg.ggml_backend_buffer_get_base(ptr);
  int getSize() => gg.ggml_backend_buffer_get_size(ptr);
  int getAlignment() => gg.ggml_backend_buffer_get_alignment(ptr);
  int getMaxSize() => gg.ggml_backend_buffer_get_max_size(ptr);
  int getAllocSize(Tensor tensor) => gg.ggml_backend_buffer_get_alloc_size(ptr, tensor.ptr);
  bool isHost() => gg.ggml_backend_buffer_is_host(ptr);
  BackendBufferType getType() => BackendBufferType.fromPtr(gg.ggml_backend_buffer_get_type(ptr));

  void initTensor(Tensor tensor) => gg.ggml_backend_buffer_init_tensor(ptr, tensor.ptr);
  void clear(int value) => gg.ggml_backend_buffer_clear(ptr, value);
  void setUsage(int usage) => gg.ggml_backend_buffer_set_usage(ptr, usage);
  void reset() => gg.ggml_backend_buffer_reset(ptr);

  void tensorAlloc(Tensor tensor, ffi.Pointer<ffi.Void> addr) =>
      gg.ggml_backend_tensor_alloc(ptr, tensor.ptr, addr);
  void viewInit(Tensor tensor) => gg.ggml_backend_view_init(ptr, tensor.ptr);

  static final finalizer =
      ggFinalizer<ffi.Pointer<gg.ggml_backend_buffer>>(ffi.Native.addressOf(gg.ggml_backend_buffer_free));
}

class BackendBufferType extends GGBase<gg.ggml_backend_buffer_type> {
  BackendBufferType.fromPtr(super.ptr) : super.fromPtr();

  BackendBuffer allocBuffer(int size) {
    final p = gg.ggml_backend_buft_alloc_buffer(ptr, size);
    return BackendBuffer.fromPtr(p);
  }

  String getName() => gg.ggml_backend_buft_name(ptr).cast<Utf8>().toDartString();
  int getAlignment() => gg.ggml_backend_buft_get_alignment(ptr);
  int getMaxSize() => gg.ggml_backend_buft_get_max_size(ptr);
  int getAllocSize(Tensor tensor) => gg.ggml_backend_buft_get_alloc_size(ptr, tensor.ptr);
  bool supportsBackend(Backend backend) => gg.ggml_backend_buft_supports_backend(ptr, backend.ptr);
  bool isHost() => gg.ggml_backend_buft_is_host(ptr);

  @override
  List<String> get props => [getName()];

  @override
  String toString() {
    return "BackendBufferType(name=${getName()})";
  }
}

class BackendGraphPlan implements ffi.Finalizable {
  BackendGraphPlan.fromPtr(this.ptr);

  gg.ggml_backend_graph_plan_t ptr;
  void dispose(Backend backend) {
    gg.ggml_backend_graph_plan_free(backend.ptr, ptr);
  }
}

class BackendEvent extends GGBase<gg.ggml_backend_event> {
  BackendEvent.fromPtr(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory BackendEvent(Backend backend) => BackendEvent.fromPtr(gg.ggml_backend_event_new(backend.ptr));

  void record() => gg.ggml_backend_event_record(ptr);
  void synchronize() => gg.ggml_backend_event_synchronize(ptr);
  void wait(Backend backend) => gg.ggml_backend_event_wait(backend.ptr, ptr);

  static final finalizer =
      ggFinalizer<gg.ggml_backend_event_t>(ffi.Native.addressOf(gg.ggml_backend_event_free));
}

class Backend extends GGBase<gg.ggml_backend> {
  Backend.fromPtr(super.ptr) : super.fromPtr();
  factory Backend([int type = GGML_BACKEND_TYPE_CPU]) {
    switch (type) {
      case GGML_BACKEND_TYPE_CPU:
        return Backend.fromPtr(gg.ggml_backend_cpu_init());
      case GGML_BACKEND_TYPE_GPU:
        throw UnimplementedError();
      case GGML_BACKEND_TYPE_GPU_SPLIT:
        throw UnimplementedError();
      default:
        throw UnimplementedError();
    }
  }

  factory Backend.cpuInit() => Backend.fromPtr(gg.ggml_backend_cpu_init());

  BackendBufferType getDefaultBufferType() {
    final p = gg.ggml_backend_get_default_buffer_type(ptr);
    return BackendBufferType.fromPtr(p);
  }

  bool isCpu() => gg.ggml_backend_is_cpu(ptr);
  List<int> guid() {
    final p = gg.ggml_backend_guid(ptr);
    return p.value.asTypedList(16);
  }

  String name() => gg.ggml_backend_name(ptr).cast<Utf8>().toDartString();

  BackendBuffer allocBuffer(int size) => BackendBuffer.fromPtr(gg.ggml_backend_alloc_buffer(ptr, size));

  int getAlignment() => gg.ggml_backend_get_alignment(ptr);
  int getMaxSize() => gg.ggml_backend_get_max_size(ptr);

  void tensorSetAsync(Tensor tensor, ffi.Pointer<ffi.Void> data, int offset, int size) =>
      gg.ggml_backend_tensor_set_async(ptr, tensor.ptr, data, offset, size);
  void tensorGetAsync(Tensor tensor, ffi.Pointer<ffi.Void> data, int offset, int size) =>
      gg.ggml_backend_tensor_get_async(ptr, tensor.ptr, data, offset, size);

  // void tensorGet(Tensor tensor, ffi.Pointer<ffi.Void> data, int offset, int size) =>
  //     gg.ggml_backend_tensor_get(tensor.ptr, data, offset, size);
  // void tensorSet(Tensor tensor, ffi.Pointer<ffi.Void> data, int offset, int size) =>
  //     gg.ggml_backend_tensor_set(tensor.ptr, data, offset, size);

  List<num> tensorGet(Tensor tensor) => tensor.tensorGet();
  void tensorSet(Tensor tensor, List<num> data) => tensor.tensorSet(data);

  void synchronize() => gg.ggml_backend_synchronize(ptr);

  BackendGraphPlan graphPlanCreate(CGraph cgraph) =>
      BackendGraphPlan.fromPtr(gg.ggml_backend_graph_plan_create(ptr, cgraph.ptr));
  void graphPlanFree(BackendGraphPlan plan) => gg.ggml_backend_graph_plan_free(ptr, plan.ptr);
  int graphPlanCompute(BackendGraphPlan plan) => gg.ggml_backend_graph_plan_compute(ptr, plan.ptr);
  int graphCompute(CGraph cgraph) => gg.ggml_backend_graph_compute(ptr, cgraph.ptr);
  int graphComputeAsync(CGraph cgraph) => gg.ggml_backend_graph_compute_async(ptr, cgraph.ptr);
  bool supportsOp(Tensor op) => gg.ggml_backend_supports_op(ptr, op.ptr);
  bool offloadOp(Tensor op) => gg.ggml_backend_offload_op(ptr, op.ptr);

  /// tensor copy between different backends
  void tensorCopy(Tensor src, Tensor dst) => gg.ggml_backend_tensor_copy(src.ptr, dst.ptr);

  /// asynchronous copy
  /// the copy is performed after all the currently queued operations in backend_src
  /// backend_dst will wait for the copy to complete before performing other operations
  /// automatic fallback to sync copy if async is not supported
  void tensorCopyAsync(Backend backendSrc, Backend backendDst, Tensor src, Tensor dst) =>
      gg.ggml_backend_tensor_copy_async(backendSrc.ptr, backendDst.ptr, src.ptr, dst.ptr);

  void cpuSetNThreads(int nThreads) {
    if (isCpu()) {
      gg.ggml_backend_cpu_set_n_threads(ptr, nThreads);
    } else {
      throw Exception("Backend is not cpu");
    }
  }

  /// Create a backend buffer from an existing pointer
  BackendBuffer cpuBufferFromPtr(ffi.Pointer<ffi.Void> ptr, int size) =>
      BackendBuffer.fromPtr(gg.ggml_backend_cpu_buffer_from_ptr(ptr, size));
  BackendBufferType cpuBufferType() => BackendBufferType.fromPtr(gg.ggml_backend_cpu_buffer_type());
// GGMLBackendBufferType cpuHbmBufferType => GGMLBackendBufferType.fromPtr(gg.ggml_backend_cpu_hbm_buffer_type());

  ///
  /// Backend registry
  ///
  /// The backend registry is a registry of all the available backends, and allows initializing backends in a generic way
  static int regGetCount() => gg.ggml_backend_reg_get_count();
  static int regFindByName(String name) {
    return using<int>((arena) {
      final cname = name.toNativeUtf8(allocator: arena);
      return gg.ggml_backend_reg_find_by_name(cname.cast());
    });
  }

  /// str is name[:params]
  static Backend regInitBackendFromStr(String backendStr) {
    return using<Backend>((arena) {
      final cbackendStr = backendStr.toNativeUtf8(allocator: arena);
      return Backend.fromPtr(gg.ggml_backend_reg_init_backend_from_str(cbackendStr.cast()));
    });
  }

  static String regGetName(int index) {
    final cname = gg.ggml_backend_reg_get_name(index);
    return cname.cast<Utf8>().toDartString();
  }

  /// params is backend-specific
  static Backend regInitBackend(int i, [String? params]) {
    return using<Backend>((arena) {
      final cparams = params == null ? ffi.nullptr : params.toNativeUtf8(allocator: arena).cast<ffi.Char>();
      return Backend.fromPtr(gg.ggml_backend_reg_init_backend(i, cparams));
    });
  }

  static BackendBufferType regGetDefaultBufferType(int i) =>
      BackendBufferType.fromPtr(gg.ggml_backend_reg_get_default_buffer_type(i));
  static BackendBuffer regAllocBuffer(int i, int size) =>
      BackendBuffer.fromPtr(gg.ggml_backend_reg_alloc_buffer(i, size));

  // TODO
  // void cpu_set_abort_callback()

  static final finalizer =
      ggFinalizer<ffi.Pointer<gg.ggml_backend>>(ffi.Native.addressOf(gg.ggml_backend_free));
}

//
// Backend scheduler
//

/// The backend scheduler allows for multiple backends to be used together
/// Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
/// The backends are selected based on:
/// - the backend that supports the operation
/// - the location of the pre-allocated tensors (e.g. the weights)
///
///  Example usage:
///
///    // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
///    // preferrably to run on the same backend as the buffer
///    ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
///
///    sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, GGML_DEFAULT_GRAPH_SIZE, false);
///
///    // initialize buffers from a max size graph (optional)
///    reserve_graph = build_graph(sched, max_batch_size);
///
///    // manually assign nodes to a backend (optional, should not be needed in most cases)
///    struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
///    ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);
///
///    ggml_backend_sched_reserve(sched, reserve_graph);
///
///    // compute
///    graph = build_graph(sched);
///    ggml_backend_sched_graph_compute(sched, graph);
///
///    // if there are graph inputs:
///    ggml_backend_sched_reset(sched);
///    ggml_backend_sched_alloc_graph(sched, graph);
///    ggml_backend_tensor_set(input_tensor, ...);
///    ggml_backend_sched_graph_compute(sched, graph);
class BackendSched extends GGBase<gg.ggml_backend_sched> {
  BackendSched.fromPtr(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory BackendSched(List<Backend> backends, List<BackendBufferType> bufts, int graphSize, bool parallel) {
    assert(backends.length == bufts.length);
    final cbackends = calloc<ffi.Pointer<gg.ggml_backend>>(backends.length);
    for (var i = 0; i < backends.length; i++) {
      cbackends[i] = backends[i].ptr;
    }
    final cbufts = calloc<ffi.Pointer<gg.ggml_backend_buffer_type>>(bufts.length);
    for (var i = 0; i < bufts.length; i++) {
      cbufts[i] = bufts[i].ptr;
    }

    final p = gg.ggml_backend_sched_new(cbackends, cbufts, backends.length, graphSize, parallel);
    return BackendSched.fromPtr(p);
  }

  /// Initialize backend buffers from a measure graph
  bool reserve(CGraph measureGraph) => gg.ggml_backend_sched_reserve(ptr, measureGraph.ptr);

  /// Get the number of splits of the last graph
  int getNSplits() => gg.ggml_backend_sched_get_n_splits(ptr);
  int getNCopies() => gg.ggml_backend_sched_get_n_copies(ptr);

  int getBufferSize(Backend backend) => gg.ggml_backend_sched_get_buffer_size(ptr, backend.ptr);

  void setTensorBackend(Tensor node, Backend backend) =>
      gg.ggml_backend_sched_set_tensor_backend(ptr, node.ptr, backend.ptr);
  Backend getTensorBackend(Tensor node) =>
      Backend.fromPtr(gg.ggml_backend_sched_get_tensor_backend(ptr, node.ptr));

  /// Allocate and compute graph on the backend scheduler
  bool allocGraph(CGraph graph) => gg.ggml_backend_sched_alloc_graph(ptr, graph.ptr);
  int graphCompute(CGraph graph) => gg.ggml_backend_sched_graph_compute(ptr, graph.ptr);
  int graphComputeAsync(CGraph graph) => gg.ggml_backend_sched_graph_compute_async(ptr, graph.ptr);
  void synchronize() => gg.ggml_backend_sched_synchronize(ptr);

  /// Reset all assignments and allocators - must be called before changing the node backends
  void reset() => gg.ggml_backend_sched_reset(ptr);

  /// Set a callback to be called for each resulting node during graph compute
  // void setEvalCallback(ggml_backend_sched_eval_callback callback, void * user_data){}

  static final finalizer =
      ggFinalizer<ffi.Pointer<gg.ggml_backend_sched>>(ffi.Native.addressOf(gg.ggml_backend_sched_free));
}

///
/// Utils
///
/// Copy a graph to a different backend
class BackendGraphCopy implements ffi.Finalizable {
  BackendGraphCopy(Backend backend, CGraph graph) : ref = gg.ggml_backend_graph_copy1(backend.ptr, graph.ptr);

  gg.ggml_backend_graph_copy ref;

  void dispose() {
    gg.ggml_backend_graph_copy_free(ref);
  }
}

int ggmlBackendRegGetCount() {
  return gg.ggml_backend_reg_get_count();
}

  /// Compare the output of two backends
// bool compareGraphBackend(GGMLBackend backend1, GGMLBackend backend2, GGMLCGraph graph, gg.ggml_backend_eval_callback callback, void * user_data);
