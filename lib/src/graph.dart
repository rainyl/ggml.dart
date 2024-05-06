import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'context.dart';
import 'tensor.dart';
import 'ggml.g.dart' as gg;

// computation graph
class CGraph extends GGStruct<gg.ggml_cgraph> {
  CGraph.fromPtr(super.ptr) : super.fromPtr();

  factory CGraph(Context ctx, {int? size, bool? grads}) {
    final p = size == null || grads == null
        ? gg.ggml_new_graph(ctx.ptr)
        : gg.ggml_new_graph_custom(ctx.ptr, size, grads);
    return CGraph.fromPtr(p);
  }

  factory CGraph.import(String fname, Context ctxData, Context ctxEval) {
    final p = using((arena) {
      final cname = fname.toNativeUtf8(allocator: arena);
      return gg.ggml_graph_import(
        cname.cast(),
        ffi.Pointer.fromAddress(ctxData.ptr.address),
        ffi.Pointer.fromAddress(ctxEval.ptr.address),
      );
    });
    return CGraph.fromPtr(p);
  }

  gg.ggml_cgraph view(int i0, int i1) => gg.ggml_graph_view(ptr, i0, i1);

  CGraph cpy() {
    final p = calloc<gg.ggml_cgraph>();
    gg.ggml_graph_cpy(ptr, p);
    return CGraph.fromPtr(p);
  }

  void reset() => gg.ggml_graph_reset(ptr);
  void clear() => gg.ggml_graph_clear(ptr);
  // print info and performance information for the graph
  void printInfo() => gg.ggml_graph_print(ptr);

  int overhead() => gg.ggml_graph_overhead();
  int overheadCustom(int size, bool grads) => gg.ggml_graph_overhead_custom(size, grads);

  // ggml_graph_plan() has to be called before ggml_graph_compute()
  // when plan.work_size > 0, caller must allocate memory for plan.work_data
  CPlan plan([int nThreads = gg.GGML_DEFAULT_N_THREADS]) => CPlan(this, nThreads);
  int compute(CPlan plan) => gg.ggml_graph_compute(ptr, plan.ptr);
  // same as ggml_graph_compute() but the work data is allocated as a part of the context
  // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
  // ggml_status
  int computeWithCtx(Context ctx, int nThreads) => gg.ggml_graph_compute_with_ctx(ctx.ptr, ptr, nThreads);

  Tensor getTensor(String name) {
    return using<Tensor>((arena) {
      final cname = name.toNativeUtf8(allocator: arena);
      final p = gg.ggml_graph_get_tensor(ptr, cname.cast());
      return Tensor.fromPtr(p);
    });
  }

  void export(String fname) {
    using<void>((arena) {
      final cname = fname.toNativeUtf8(allocator: arena);
      gg.ggml_graph_export(ptr, cname.cast());
    });
  }

  // dump the graph into a file using the dot format
  void dumpDot(CGraph gb, CGraph gf, String filename) {
    using<void>((arena) {
      final cname = filename.toNativeUtf8(allocator: arena);
      gg.ggml_graph_dump_dot(gb.ptr, gf.ptr, cname.cast());
    });
  }

  // build_forward_expand
  void buildForwardExpand(Tensor tensor) {
    gg.ggml_build_forward_expand(ptr, tensor.ptr);
  }

  gg.ggml_cgraph get reg => ptr.ref;
}

class CPlan extends GGStruct<gg.ggml_cplan> {
  CPlan.fromPtr(super.ptr) : super.fromPtr();

  factory CPlan(CGraph graph, [int nThreads = gg.GGML_DEFAULT_N_THREADS]) {
    final s = gg.ggml_graph_plan(graph.ptr, nThreads);
    final p = calloc<gg.ggml_cplan>()..ref = s;
    return CPlan.fromPtr(p);
  }

  @override
  gg.ggml_cplan get ref => ptr.ref;
}
