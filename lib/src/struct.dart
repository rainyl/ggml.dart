// ignore_for_file: non_constant_identifier_names

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'backend.dart';
import 'alloc.dart';
import 'base.dart';
import 'ggml.g.dart' as gg;

class GGMLInitParams extends GgStruct<gg.ggml_init_params> {
  GGMLInitParams._(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory GGMLInitParams({int? memSize, bool? noAlloc, ffi.Pointer<ffi.Void>? memBuffer}) {
    final p = calloc<gg.ggml_init_params>()
      ..ref.mem_size = memSize ?? 0
      ..ref.mem_buffer = memBuffer ?? ffi.nullptr
      ..ref.no_alloc = noAlloc ?? false;
    return GGMLInitParams._(p);
  }

  int get memSize => ref.mem_size;
  bool get noAlloc => ref.no_alloc;
  ffi.Pointer<ffi.Void> get memBuffer => ref.mem_buffer;

  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  @override
  gg.ggml_init_params get ref => ptr.ref;
}

class GGMLObject extends GgStruct<gg.ggml_object> {
  GGMLObject._(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory GGMLObject(int offs, int size, GGMLObject next, int type, List<int> padding) {
    assert(padding.length == 4);
    final pPadding = ffi.Array<ffi.Char>(padding.length);
    for (int i = 0; i < padding.length; i++) {
      pPadding[i] = padding[i];
    }
    final p = calloc<gg.ggml_object>()
      ..ref.offs = offs
      ..ref.size = size
      ..ref.type = type
      ..ref.next = next.ptr
      ..ref.padding = pPadding;
    return GGMLObject._(p);
  }

  int get offs => ref.offs;
  int get size => ref.size;
  int get type => ref.type;

  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  @override
  gg.ggml_object get ref => ptr.ref;
}

class GGMLCPlan extends GgStruct<gg.ggml_cplan> {
  GGMLCPlan._(super.ptr) : super.fromPtr();

  factory GGMLCPlan(GGMLCGraph graph, [int nThreads = gg.GGML_DEFAULT_N_THREADS]) {
    final s = gg.ggml_graph_plan(graph.ptr, nThreads);
    final p = calloc<gg.ggml_cplan>()..ref = s;
    return GGMLCPlan._(p);
  }

  @override
  gg.ggml_cplan get ref => ptr.ref;
}

// computation graph
class GGMLCGraph extends GgStruct<gg.ggml_cgraph> {
  GGMLCGraph._(super.ptr) : super.fromPtr();

  factory GGMLCGraph(GGMLContext ctx, {int? size, bool? grads}) {
    final p = size == null || grads == null
        ? gg.ggml_new_graph(ctx.ptr)
        : gg.ggml_new_graph_custom(ctx.ptr, size, grads);
    return GGMLCGraph._(p);
  }

  factory GGMLCGraph.import(String fname, GGMLContext ctxData, GGMLContext ctxEval) {
    final p = using((arena) {
      final cname = fname.toNativeUtf8(allocator: arena);
      return gg.ggml_graph_import(
        cname.cast(),
        ffi.Pointer.fromAddress(ctxData.ptr.address),
        ffi.Pointer.fromAddress(ctxEval.ptr.address),
      );
    });
    return GGMLCGraph._(p);
  }

  gg.ggml_cgraph view(int i0, int i1) => gg.ggml_graph_view(ptr, i0, i1);

  GGMLCGraph cpy() {
    final p = calloc<gg.ggml_cgraph>();
    gg.ggml_graph_cpy(ptr, p);
    return GGMLCGraph._(p);
  }

  void reset() => gg.ggml_graph_reset(ptr);
  void clear() => gg.ggml_graph_clear(ptr);
  // print info and performance information for the graph
  void print() => gg.ggml_graph_print(ptr);

  int overhead() => gg.ggml_graph_overhead();
  int overheadCustom(int size, bool grads) => gg.ggml_graph_overhead_custom(size, grads);

  // ggml_graph_plan() has to be called before ggml_graph_compute()
  // when plan.work_size > 0, caller must allocate memory for plan.work_data
  GGMLCPlan plan([int nThreads = gg.GGML_DEFAULT_N_THREADS]) => GGMLCPlan(this, nThreads);
  int compute(GGMLCPlan plan) => gg.ggml_graph_compute(ptr, plan.ptr);
  // same as ggml_graph_compute() but the work data is allocated as a part of the context
  // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
  int computeWithCtx(GGMLContext ctx, int nThreads) => gg.ggml_graph_compute_with_ctx(ctx.ptr, ptr, nThreads);

  GGMLTensor getTensor(String name) {
    return using<GGMLTensor>((arena) {
      final cname = name.toNativeUtf8(allocator: arena);
      final p = gg.ggml_graph_get_tensor(ptr, cname.cast());
      return GGMLTensor._(p);
    });
  }

  void export(String fname) {
    using<void>((arena) {
      final cname = fname.toNativeUtf8(allocator: arena);
      gg.ggml_graph_export(ptr, cname.cast());
    });
  }

  // dump the graph into a file using the dot format
  void dumpDot(GGMLCGraph gb, GGMLCGraph gf, String filename) {
    using<void>((arena) {
      final cname = filename.toNativeUtf8(allocator: arena);
      gg.ggml_graph_dump_dot(gb.ptr, gf.ptr, cname.cast());
    });
  }

  gg.ggml_cgraph get reg => ptr.ref;
}

class GGMLScratch extends GgStruct<gg.ggml_scratch> {
  GGMLScratch._(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory GGMLScratch({int? offs, int? size, ffi.Pointer<ffi.Void>? data}) {
    final p = calloc<gg.ggml_scratch>()
      ..ref.offs = offs ?? 0
      ..ref.size = size ?? 0
      ..ref.data = data ?? ffi.nullptr;
    return GGMLScratch._(p);
  }

  int get offs => ref.offs;
  int get size => ref.size;
  ffi.Pointer<ffi.Void> get data => ref.data;

  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  @override
  gg.ggml_scratch get ref => ptr.ref;
}

class GGMLContext extends GgObject<gg.ggml_context> {
  GGMLContext._(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory GGMLContext.init(GGMLInitParams params) {
    final p = gg.ggml_init(params.ref);
    return GGMLContext._(p);
  }

  GGMLTensor newTensor(int type, List<int> ne) {
    final pne = calloc<ffi.Int64>(ne.length);
    for (int i = 0; i < ne.length; i++) {
      pne[i] = ne[i];
    }
    final p = gg.ggml_new_tensor(ptr, type, ne.length, pne);
    return GGMLTensor._(p);
  }

  GGMLTensor newTensor1D(int type, int ne0) {
    final p = gg.ggml_new_tensor_1d(ptr, type, ne0);
    return GGMLTensor._(p);
  }

  GGMLTensor newTensor2D(int type, int ne0, int ne1) {
    final p = gg.ggml_new_tensor_2d(ptr, type, ne0, ne1);
    return GGMLTensor._(p);
  }

  GGMLTensor newTensor3D(int type, int ne0, int ne1, int ne2) {
    final p = gg.ggml_new_tensor_3d(ptr, type, ne0, ne1, ne2);
    return GGMLTensor._(p);
  }

  GGMLTensor newTensor4D(int type, int ne0, int ne1, int ne2, int ne3) {
    final p = gg.ggml_new_tensor_4d(ptr, type, ne0, ne1, ne2, ne3);
    return GGMLTensor._(p);
  }

  GGMLTensor dupTensor(GGMLTensor src) {
    final p = gg.ggml_dup_tensor(ptr, src.ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor viewTensor(GGMLTensor src) {
    final p = gg.ggml_view_tensor(ptr, src.ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor getFirstTensor() {
    final p = gg.ggml_get_first_tensor(ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor getNextTensor(GGMLTensor src) {
    final p = gg.ggml_get_next_tensor(ptr, src.ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor getTensor(String name) {
    return using<GGMLTensor>((arena) {
      final cname = name.toNativeUtf8(allocator: arena);
      final p = gg.ggml_get_next_tensor(ptr, cname.cast());
      return GGMLTensor._(p);
    });
  }

  GGMLTensor dup(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_dup_inplace(ptr, a.ptr) : gg.ggml_dup(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor add(GGMLTensor a, GGMLTensor b, [bool inplace = false, bool cast = false, int? castType]) {
    if (cast && castType != null) {
      return GGMLTensor._(gg.ggml_add_cast(ptr, a.ptr, b.ptr, castType));
    }
    final p = inplace ? gg.ggml_add_inplace(ptr, a.ptr, b.ptr) : gg.ggml_add(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor add1(GGMLTensor a, GGMLTensor b, [bool inplace = false]) {
    final p = inplace ? gg.ggml_add1_inplace(ptr, a.ptr, b.ptr) : gg.ggml_add1(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor acc(GGMLTensor a, GGMLTensor b, int nb1, int nb2, int nb3, int offset, [bool inplace = false]) {
    final p = inplace
        ? gg.ggml_acc_inplace(ptr, a.ptr, b.ptr, nb1, nb2, nb3, offset)
        : gg.ggml_acc(ptr, a.ptr, b.ptr, nb1, nb2, nb3, offset);
    return GGMLTensor._(p);
  }

  GGMLTensor sub(GGMLTensor a, GGMLTensor b, [bool inplace = false]) {
    final p = inplace ? gg.ggml_sub_inplace(ptr, a.ptr, b.ptr) : gg.ggml_sub(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor mul(GGMLTensor a, GGMLTensor b, [bool inplace = false]) {
    final p = inplace ? gg.ggml_mul_inplace(ptr, a.ptr, b.ptr) : gg.ggml_mul(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor div(GGMLTensor a, GGMLTensor b, [bool inplace = false]) {
    final p = inplace ? gg.ggml_div_inplace(ptr, a.ptr, b.ptr) : gg.ggml_div(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor sqr(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_sqr_inplace(ptr, a.ptr) : gg.ggml_sqr(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor sqrt(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_sqrt_inplace(ptr, a.ptr) : gg.ggml_sqrt(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor log(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_log_inplace(ptr, a.ptr) : gg.ggml_log(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  /// sum
  /// return scalar
  GGMLTensor sum(GGMLTensor a) {
    final p = gg.ggml_sum(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  /// sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
  GGMLTensor sumRows(GGMLTensor a) {
    final p = gg.ggml_sum_rows(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  /// mean along rows
  GGMLTensor mean(GGMLTensor a) {
    final p = gg.ggml_mean(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  /// argmax along rows
  GGMLTensor argmax(GGMLTensor a) {
    final p = gg.ggml_argmax(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // if a is the same shape as b, and a is not parameter, return a
  // otherwise, return a new tensor: repeat(a) to fit in b
  GGMLTensor repeat(GGMLTensor a, GGMLTensor b) {
    final p = gg.ggml_repeat(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  // sums repetitions in a into shape of b
  GGMLTensor repeatBack(GGMLTensor a, GGMLTensor b) {
    final p = gg.ggml_repeat_back(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  // concat a and b on dim 2
  // used in stable-diffusion
  GGMLTensor concat(GGMLTensor a, GGMLTensor b) {
    final p = gg.ggml_concat(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  // abs
  GGMLTensor abs(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_abs_inplace(ptr, a.ptr) : gg.ggml_abs(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // sgn
  GGMLTensor sgn(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_sgn_inplace(ptr, a.ptr) : gg.ggml_sgn(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // neg
  GGMLTensor neg(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_neg_inplace(ptr, a.ptr) : gg.ggml_neg(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  //tanh
  GGMLTensor tanh(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_tanh_inplace(ptr, a.ptr) : gg.ggml_tanh(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // elu
  GGMLTensor elu(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_elu_inplace(ptr, a.ptr) : gg.ggml_elu(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // relu
  GGMLTensor relu(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_relu_inplace(ptr, a.ptr) : gg.ggml_relu(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // leaky_relu
  GGMLTensor leakyRelu(GGMLTensor a, double negative_slope, [bool inplace = false]) {
    final p = gg.ggml_leaky_relu(ptr, a.ptr, negative_slope, inplace);
    return GGMLTensor._(p);
  }

  // gelu
  GGMLTensor gelu(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_gelu_inplace(ptr, a.ptr) : gg.ggml_gelu(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // gelu_quick
  GGMLTensor geluQuick(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_gelu_quick_inplace(ptr, a.ptr) : gg.ggml_gelu_quick(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // silu
  GGMLTensor silu(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_silu_inplace(ptr, a.ptr) : gg.ggml_silu(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // a - x
  // b - dy
  GGMLTensor siluBack(GGMLTensor a, GGMLTensor b) {
    final p = gg.ggml_silu_back(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  // hardswish(x) = x * relu6(x + 3) / 6
  GGMLTensor hardswish(GGMLTensor a) {
    final p = gg.ggml_hardswish(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // hardsigmoid(x) = relu6(x + 3) / 6
  GGMLTensor hardsigmoid(GGMLTensor a) {
    final p = gg.ggml_hardsigmoid(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // normalize along rows
  GGMLTensor norm(GGMLTensor a, double eps, [bool inplace = false]) {
    final p = inplace ? gg.ggml_norm_inplace(ptr, a.ptr, eps) : gg.ggml_norm(ptr, a.ptr, eps);
    return GGMLTensor._(p);
  }

  // rms_norm
  GGMLTensor rmsNorm(GGMLTensor a, double eps, [bool inplace = false]) {
    final p = inplace ? gg.ggml_rms_norm_inplace(ptr, a.ptr, eps) : gg.ggml_rms_norm(ptr, a.ptr, eps);
    return GGMLTensor._(p);
  }

  // group normalize along ne0*ne1*n_groups
  // used in stable-diffusion
  // TODO: eps is hardcoded to 1e-6 for now
  GGMLTensor groupNorm(GGMLTensor a, int n_groups, [bool inplace = false]) {
    final p =
        inplace ? gg.ggml_group_norm_inplace(ptr, a.ptr, n_groups) : gg.ggml_group_norm(ptr, a.ptr, n_groups);
    return GGMLTensor._(p);
  }

  // a - x
  // b - dy
  GGMLTensor rmsNormBack(GGMLTensor a, GGMLTensor b, double eps) {
    final p = gg.ggml_rms_norm_back(ptr, a.ptr, b.ptr, eps);
    return GGMLTensor._(p);
  }

  // A: k columns, n rows => [ne03, ne02, n, k]
  // B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
  // result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
  GGMLTensor mulMat(GGMLTensor a, GGMLTensor b) {
    final p = gg.ggml_mul_mat(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  // change the precision of a matrix multiplication
  // set to GGML_PREC_F32 for higher precision (useful for phi-2)
  void mulMatSetPrec(GGMLTensor a, int prec) {
    gg.ggml_mul_mat_set_prec(a.ptr, prec);
  }

  // indirect matrix multiplication
  //  ggml_mul_mat_id(ctx, as, ids, id, b) ~= ggml_mul_mat(as[ids[id]], b)
  GGMLTensor mulMatId(GGMLTensor as, GGMLTensor ids, int id, GGMLTensor b) {
    final p = gg.ggml_mul_mat_id(ptr, as.ptr, ids.ptr, id, b.ptr);
    return GGMLTensor._(p);
  }

  // A: m columns, n rows,
  // B: p columns, n rows,
  // result is m columns, p rows
  GGMLTensor outProd(GGMLTensor a, GGMLTensor b) {
    final p = gg.ggml_out_prod(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  //
  // operations on tensors without backpropagation
  //

  GGMLTensor scale(GGMLTensor a, double s, [bool inplace = false]) {
    final p = inplace ? gg.ggml_scale_inplace(ptr, a.ptr, s) : gg.ggml_scale(ptr, a.ptr, s);
    return GGMLTensor._(p);
  }

  // set
  GGMLTensor set(GGMLTensor a, GGMLTensor b, int nb1, int nb2, int nb3, int offset, [bool inplace = false]) {
    final p = inplace
        ? gg.ggml_set_inplace(ptr, a.ptr, b.ptr, nb1, nb2, nb3, offset)
        : gg.ggml_set(ptr, a.ptr, b.ptr, nb1, nb2, nb3, offset);
    return GGMLTensor._(p);
  }

  GGMLTensor set1D(GGMLTensor a, GGMLTensor b, int offset, [bool inplace = false]) {
    final p = inplace
        ? gg.ggml_set_1d_inplace(ptr, a.ptr, b.ptr, offset)
        : gg.ggml_set_1d(ptr, a.ptr, b.ptr, offset);
    return GGMLTensor._(p);
  }

  /// b -> view(a,offset,nb1,nb2,3)
  GGMLTensor set2D(GGMLTensor a, GGMLTensor b, int nb1, int offset, [bool inplace = false]) {
    final p = inplace
        ? gg.ggml_set_2d_inplace(ptr, a.ptr, b.ptr, nb1, offset)
        : gg.ggml_set_2d(ptr, a.ptr, b.ptr, nb1, offset);
    return GGMLTensor._(p);
  }

  /// a -> b, return view(b)
  GGMLTensor cpy(GGMLTensor a, GGMLTensor b) {
    final p = gg.ggml_cpy(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  // cast
  GGMLTensor cast(GGMLTensor a, int type) {
    final p = gg.ggml_cast(ptr, a.ptr, type);
    return GGMLTensor._(p);
  }

  // make contiguous
  GGMLTensor cont(GGMLTensor a) {
    final p = gg.ggml_cont(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // make contiguous, with new shape
  GGMLTensor cont1D(GGMLTensor a, int ne0) {
    final p = gg.ggml_cont_1d(ptr, a.ptr, ne0);
    return GGMLTensor._(p);
  }

  GGMLTensor cont2D(GGMLTensor a, int ne0, int ne1) {
    final p = gg.ggml_cont_2d(ptr, a.ptr, ne0, ne1);
    return GGMLTensor._(p);
  }

  GGMLTensor cont3D(GGMLTensor a, int ne0, int ne1, int ne2) {
    final p = gg.ggml_cont_3d(ptr, a.ptr, ne0, ne1, ne2);
    return GGMLTensor._(p);
  }

  GGMLTensor cont4D(GGMLTensor a, int ne0, int ne1, int ne2, int ne3) {
    final p = gg.ggml_cont_4d(ptr, a.ptr, ne0, ne1, ne2, ne3);
    return GGMLTensor._(p);
  }

  // return view(a), b specifies the new shape
  // TODO: when we start computing gradient, make a copy instead of view
  GGMLTensor reshape(GGMLTensor a, GGMLTensor b) {
    final p = gg.ggml_reshape(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  // return view(a)
  // TODO: when we start computing gradient, make a copy instead of view
  GGMLTensor reshape1D(GGMLTensor a, int ne0) {
    final p = gg.ggml_reshape_1d(ptr, a.ptr, ne0);
    return GGMLTensor._(p);
  }

  GGMLTensor reshape2D(GGMLTensor a, int ne0, int ne1) {
    final p = gg.ggml_reshape_2d(ptr, a.ptr, ne0, ne1);
    return GGMLTensor._(p);
  }

  GGMLTensor reshape3D(GGMLTensor a, int ne0, int ne1, int ne2) {
    final p = gg.ggml_reshape_3d(ptr, a.ptr, ne0, ne1, ne2);
    return GGMLTensor._(p);
  }

  GGMLTensor reshape4D(GGMLTensor a, int ne0, int ne1, int ne2, int ne3) {
    final p = gg.ggml_reshape_4d(ptr, a.ptr, ne0, ne1, ne2, ne3);
    return GGMLTensor._(p);
  }

  // offset in bytes
  GGMLTensor view1D(GGMLTensor a, int ne0, int offset) {
    final p = gg.ggml_view_1d(ptr, a.ptr, ne0, offset);
    return GGMLTensor._(p);
  }

  GGMLTensor view2D(GGMLTensor a, int ne0, int ne1, int nb1, int offset) {
    final p = gg.ggml_view_2d(ptr, a.ptr, ne0, ne1, nb1, offset);
    return GGMLTensor._(p);
  }

  GGMLTensor view3D(GGMLTensor a, int ne0, int ne1, int ne2, int nb1, int nb2, int offset) {
    final p = gg.ggml_view_3d(ptr, a.ptr, ne0, ne1, ne2, nb1, nb2, offset);
    return GGMLTensor._(p);
  }

  GGMLTensor view4D(GGMLTensor a, int ne0, int ne1, int ne2, int ne3, int nb1, int nb2, int nb3, int offset) {
    final p = gg.ggml_view_4d(ptr, a.ptr, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);
    return GGMLTensor._(p);
  }

  GGMLTensor permute(GGMLTensor a, int axis0, int axis1, int axis2, int axis3) {
    final p = gg.ggml_permute(ptr, a.ptr, axis0, axis1, axis2, axis3);
    return GGMLTensor._(p);
  }

  // alias for ggml_permute(ctx, a, 1, 0, 2, 3)
  GGMLTensor transpose(GGMLTensor a) {
    final p = gg.ggml_transpose(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // supports 3D: a->ne[2] == b->ne[1]
  GGMLTensor getRows(GGMLTensor a, GGMLTensor b) {
    final p = gg.ggml_get_rows(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor getRowsBack(GGMLTensor a, GGMLTensor b, GGMLTensor c) {
    final p = gg.ggml_get_rows_back(ptr, a.ptr, b.ptr, c.ptr);
    return GGMLTensor._(p);
  }

  // diag
  GGMLTensor diag(GGMLTensor a) {
    final p = gg.ggml_diag(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // set elements above the diagonal to -INF
  GGMLTensor diagMaskInf(GGMLTensor a, int n, [bool inplace = false]) {
    final p = inplace ? gg.ggml_diag_mask_inf_inplace(ptr, a.ptr, n) : gg.ggml_diag_mask_inf(ptr, a.ptr, n);
    return GGMLTensor._(p);
  }

  // set elements above the diagonal to 0
  GGMLTensor diagMaskZero(GGMLTensor a, int n, [bool inplace = false]) {
    final p = inplace ? gg.ggml_diag_mask_zero_inplace(ptr, a.ptr, n) : gg.ggml_diag_mask_zero(ptr, a.ptr, n);
    return GGMLTensor._(p);
  }

  // softmax
  GGMLTensor softMax(GGMLTensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_soft_max_inplace(ptr, a.ptr) : gg.ggml_soft_max(ptr, a.ptr);
    return GGMLTensor._(p);
  }

  // fused soft_max(a*scale + mask + pos[i]*(ALiBi slope))
  // mask is optional
  // pos is required when max_bias > 0.0f
  // max_bias = 0.0f for no ALiBi
  GGMLTensor softMaxExt(GGMLTensor a, GGMLTensor mask, GGMLTensor pos, double scale, double maxBias) {
    final p = gg.ggml_soft_max_ext(ptr, a.ptr, mask.ptr, pos.ptr, scale, maxBias);
    return GGMLTensor._(p);
  }

  GGMLTensor softMaxBack(GGMLTensor a, GGMLTensor b, [bool inplace = false]) {
    final p =
        inplace ? gg.ggml_soft_max_back_inplace(ptr, a.ptr, b.ptr) : gg.ggml_soft_max_back(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  // rotary position embedding
  // if mode & 1 == 1, skip n_past elements (DEPRECATED)
  // if mode & 2 == 1, GPT-NeoX style
  // if mode & 4 == 1, ChatGLM style
  //
  // b is an int32 vector with size a->ne[2], it contains the positions
  GGMLTensor rope(GGMLTensor a, GGMLTensor b, int nDims, int mode, int nCtx, [bool inplace = false]) {
    final p = inplace
        ? gg.ggml_rope_inplace(ptr, a.ptr, b.ptr, nDims, mode, nCtx)
        : gg.ggml_rope(ptr, a.ptr, b.ptr, nDims, mode, nCtx);
    return GGMLTensor._(p);
  }

  // custom RoPE
  GGMLTensor ropeCustom(
    GGMLTensor a,
    GGMLTensor b,
    int nDims,
    int mode,
    int nCtx,
    int nOrigCtx,
    double freqBase,
    double freqScale,
    double extFactor,
    double attnFactor,
    double betaFast,
    double betaSlow, {
    bool inplace = false,
  }) {
    final p = inplace
        ? gg.ggml_rope_custom_inplace(ptr, a.ptr, b.ptr, nDims, mode, nCtx, nOrigCtx, extFactor, attnFactor,
            betaFast, betaSlow, freqBase, freqScale)
        : gg.ggml_rope_custom(ptr, a.ptr, b.ptr, nDims, mode, nCtx, nOrigCtx, extFactor, attnFactor, betaFast,
            betaSlow, freqBase, freqScale);
    return GGMLTensor._(p);
  }

  /// compute correction dims for YaRN RoPE scaling
  (double, double) rope_yarn_corr_dims(
      int nDims, int nOrigCtx, double freqBase, double betaFast, double betaSlow) {
    return using<(double, double)>((arena) {
      final pres = arena<ffi.Float>(2);
      gg.ggml_rope_yarn_corr_dims(nDims, nOrigCtx, freqBase, betaFast, betaSlow, pres);
      return (pres[0], pres[1]);
    });
  }

  // xPos RoPE, in-place, returns view(a)
  GGMLTensor ropeXposInplace(GGMLTensor a, GGMLTensor b, int nDims, double base, bool down) {
    final p = gg.ggml_rope_xpos_inplace(ptr, a.ptr, b.ptr, nDims, base, down);
    return GGMLTensor._(p);
  }

  // rotary position embedding backward, i.e compute dx from dy
  // a - dy
  GGMLTensor ropeBack(
      GGMLTensor a,
      GGMLTensor b,
      int nDims,
      int mode,
      int nCtx,
      int nOrigCtx,
      double freqBase,
      double freqScale,
      double extFactor,
      double attnFactor,
      double betaFast,
      double betaSlow,
      double xPosBase,
      bool xPosDown) {
    final p = gg.ggml_rope_back(ptr, a.ptr, b.ptr, nDims, mode, nCtx, nOrigCtx, extFactor, attnFactor,
        betaFast, betaSlow, freqBase, freqScale, xPosBase, xPosDown);
    return GGMLTensor._(p);
  }

  // clamp
  GGMLTensor clamp(GGMLTensor a, double min, double max) {
    final p = gg.ggml_clamp(ptr, a.ptr, min, max);
    return GGMLTensor._(p);
  }

  // im2col
  GGMLTensor im2col(
      GGMLTensor a, GGMLTensor b, int s0, int s1, int p0, int p1, int d0, int d1, bool is2D, int dstType) {
    final p = gg.ggml_im2col(ptr, a.ptr, b.ptr, s0, s1, p0, p1, d0, d1, is2D, dstType);
    return GGMLTensor._(p);
  }

  // conv_depthwise_2d
  GGMLTensor convDepthwise2d(GGMLTensor a, GGMLTensor b, int s0, int s1, int p0, int p1, int d0, int d1) {
    final p = gg.ggml_conv_depthwise_2d(ptr, a.ptr, b.ptr, s0, s1, p0, p1, d0, d1);
    return GGMLTensor._(p);
  }

  // conv_1d
  GGMLTensor conv1d(GGMLTensor a, GGMLTensor b, int s0, int p0, int d0) {
    final p = gg.ggml_conv_1d(ptr, a.ptr, b.ptr, s0, p0, d0);
    return GGMLTensor._(p);
  }

  // conv_1d with padding = half
  // alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)
  GGMLTensor conv1dPh(GGMLTensor a, GGMLTensor b, int s, int d) {
    final p = gg.ggml_conv_1d_ph(ptr, a.ptr, b.ptr, s, d);
    return GGMLTensor._(p);
  }

  // conv_transpose_1d
  GGMLTensor convTranspose1d(GGMLTensor a, GGMLTensor b, int s0, int p0, int d0) {
    final p = gg.ggml_conv_transpose_1d(ptr, a.ptr, b.ptr, s0, p0, d0);
    return GGMLTensor._(p);
  }

  // conv_2d
  GGMLTensor conv2d(GGMLTensor a, GGMLTensor b, int s0, int s1, int p0, int p1, int d0, int d1) {
    final p = gg.ggml_conv_2d(ptr, a.ptr, b.ptr, s0, s1, p0, p1, d0, d1);
    return GGMLTensor._(p);
  }

  // kernel size is a->ne[0] x a->ne[1]
  // stride is equal to kernel size
  // padding is zero
  // example:
  // a:     16   16    3  768
  // b:   1024 1024    3    1
  // res:   64   64  768    1
  // used in sam
  // conv_2d_sk_p0
  GGMLTensor conv2dSkP0(GGMLTensor a, GGMLTensor b) {
    final p = gg.ggml_conv_2d_sk_p0(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  // kernel size is a->ne[0] x a->ne[1]
  // stride is 1
  // padding is half
  // example:
  // a:      3    3    256  256
  // b:     64   64    256    1
  // res:   64   64    256    1
  // used in sam
  // conv_2d_s1_ph
  GGMLTensor conv2dS1Ph(GGMLTensor a, GGMLTensor b) {
    final p = gg.ggml_conv_2d_s1_ph(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  // conv_transpose_2d_p0
  GGMLTensor convTranspose2dP0(GGMLTensor a, GGMLTensor b, int stride) {
    final p = gg.ggml_conv_transpose_2d_p0(ptr, a.ptr, b.ptr, stride);
    return GGMLTensor._(p);
  }

  // pool_1d
  GGMLTensor pool1d(GGMLTensor a, int op, int k0, int s0, int p0) {
    final p = gg.ggml_pool_1d(ptr, a.ptr, op, k0, s0, p0);
    return GGMLTensor._(p);
  }

  // the result will have 2*p0 padding for the first dimension
  // and 2*p1 padding for the second dimension
  GGMLTensor pool2d(GGMLTensor a, int op, int k0, int k1, int s0, int s1, double p0, double p1) {
    final p = gg.ggml_pool_2d(ptr, a.ptr, op, k0, k1, s0, s1, p0, p1);
    return GGMLTensor._(p);
  }

  // nearest interpolate
  // used in stable-diffusion
  // upscale
  GGMLTensor upscale(GGMLTensor a, int scaleFactor) {
    final p = gg.ggml_upscale(ptr, a.ptr, scaleFactor);
    return GGMLTensor._(p);
  }

  // pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]
  GGMLTensor pad(GGMLTensor a, int p0, int p1, int p2, int p3) {
    final p = gg.ggml_pad(ptr, a.ptr, p0, p1, p2, p3);
    return GGMLTensor._(p);
  }

  // Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
  // timesteps: [N,]
  // return: [N, dim]
  GGMLTensor timestepEmbedding(GGMLTensor timesteps, int dim, int maxPeriod) {
    final p = gg.ggml_timestep_embedding(ptr, timesteps.ptr, dim, maxPeriod);
    return GGMLTensor._(p);
  }

  // argsort
  GGMLTensor argsort(GGMLTensor a, int order) {
    final p = gg.ggml_argsort(ptr, a.ptr, order);
    return GGMLTensor._(p);
  }

  // arange
  GGMLTensor arange(double start, double stop, double step) {
    final p = gg.ggml_arange(ptr, start, stop, step);
    return GGMLTensor._(p);
  }

  // top k elements per row
  GGMLTensor topk(GGMLTensor a, int k) {
    final p = gg.ggml_top_k(ptr, a.ptr, k);
    return GGMLTensor._(p);
  }

  // flash_attn
  GGMLTensor flashAttn(GGMLTensor q, GGMLTensor k, GGMLTensor v, bool masked) {
    final p = gg.ggml_flash_attn(ptr, q.ptr, k.ptr, v.ptr, masked);
    return GGMLTensor._(p);
  }

  // flash_attn_back
  GGMLTensor flashAttnBack(GGMLTensor q, GGMLTensor k, GGMLTensor v, GGMLTensor d, bool masked) {
    final p = gg.ggml_flash_attn_back(ptr, q.ptr, k.ptr, v.ptr, d.ptr, masked);
    return GGMLTensor._(p);
  }

  // flash_ff
  GGMLTensor flashFf(GGMLTensor a, GGMLTensor b0, GGMLTensor b1, GGMLTensor c0, GGMLTensor c1) {
    final p = gg.ggml_flash_ff(ptr, a.ptr, b0.ptr, b1.ptr, c0.ptr, c1.ptr);
    return GGMLTensor._(p);
  }

  // ssm_conv
  GGMLTensor ssmConv(GGMLTensor s, GGMLTensor x, GGMLTensor c, GGMLTensor sq) {
    final p = gg.ggml_ssm_conv(ptr, s.ptr, x.ptr, c.ptr, sq.ptr);
    return GGMLTensor._(p);
  }

  // ssm_scan
  GGMLTensor ssmScan(
      GGMLTensor s, GGMLTensor x, GGMLTensor dt, GGMLTensor A, GGMLTensor B, GGMLTensor C, GGMLTensor sq) {
    final p = gg.ggml_ssm_scan(ptr, s.ptr, x.ptr, dt.ptr, A.ptr, B.ptr, C.ptr, sq.ptr);
    return GGMLTensor._(p);
  }

  // partition into non-overlapping windows with padding if needed
  // example:
  // a:   768   64   64    1
  // w:    14
  // res: 768   14   14    25
  // used in sam
  GGMLTensor winPart(GGMLTensor a, int w) {
    final p = gg.ggml_win_part(ptr, a.ptr, w);
    return GGMLTensor._(p);
  }

  // reverse of ggml_win_part
  // used in sam
  GGMLTensor winUnpart(GGMLTensor a, int w0, int h0, int w) {
    final p = gg.ggml_win_unpart(ptr, a.ptr, w0, h0, w);
    return GGMLTensor._(p);
  }

  // unary
  GGMLTensor unary(GGMLTensor a, int op, [bool inplace = false]) {
    final p = inplace ? gg.ggml_unary_inplace(ptr, a.ptr, op) : gg.ggml_unary(ptr, a.ptr, op);
    return GGMLTensor._(p);
  }

  // get_rel_pos
  GGMLTensor getRelPos(GGMLTensor a, int qh, int kh) {
    final p = gg.ggml_get_rel_pos(ptr, a.ptr, qh, kh);
    return GGMLTensor._(p);
  }

  // add_rel_pos
  GGMLTensor addRelPos(GGMLTensor a, GGMLTensor pw, GGMLTensor ph, [bool inplace = false]) {
    final p = inplace
        ? gg.ggml_add_rel_pos_inplace(ptr, a.ptr, pw.ptr, ph.ptr)
        : gg.ggml_add_rel_pos(ptr, a.ptr, pw.ptr, ph.ptr);
    return GGMLTensor._(p);
  }

  // custom operators v2
  // map_custom1
  // GgmlTensor mapCustom1(GgmlTensor a, int op, int nTasks, ffi.Pointer<ffi.Void> userdata){
  //   final p = gg.ggml_map_custom1(ptr, a.ptr, op, nTasks, userdata);
  // }

  // loss function
  // cross_entropy_loss
  GGMLTensor crossEntropyLoss(GGMLTensor a, GGMLTensor b) {
    final p = gg.ggml_cross_entropy_loss(ptr, a.ptr, b.ptr);
    return GGMLTensor._(p);
  }

  // cross_entropy_loss_back
  GGMLTensor crossEntropyLossBack(GGMLTensor a, GGMLTensor b, GGMLTensor c) {
    final p = gg.ggml_cross_entropy_loss_back(ptr, a.ptr, b.ptr, c.ptr);
    return GGMLTensor._(p);
  }

  //
  // automatic differentiation
  //
  // set_param
  void setParam(GGMLTensor a) {
    gg.ggml_set_param(ptr, a.ptr);
  }

  // build_forward_expand
  void buildForwardExpand(GGMLCGraph cgraph, GGMLTensor tensor) {
    gg.ggml_build_forward_expand(cgraph.ptr, tensor.ptr);
  }

  // build_backward_expand
  void buildBackwardExpand(GGMLCGraph gf, GGMLCGraph gb, bool keep) {
    gg.ggml_build_backward_expand(ptr, gf.ptr, gb.ptr, keep);
  }

  // graph allocation in a context
  GGMLCGraph newGraph() {
    final p = gg.ggml_new_graph(ptr);
    return GGMLCGraph._(p);
  }

  GGMLCGraph newGraphCustom(int size, bool grads) {
    final p = gg.ggml_new_graph_custom(ptr, size, grads);
    return GGMLCGraph._(p);
  }

  GGMLCGraph graphDup(GGMLCGraph cgraph) {
    final p = gg.ggml_graph_dup(ptr, cgraph.ptr);
    return GGMLCGraph._(p);
  }

  // build gradient checkpointing backward graph gb for gf using provided checkpoints
  // gb_tmp will contain original backward graph with rewritten backward process nodes,
  // but without the second forward pass nodes.
  void buildBackwardGradientCheckpointing(
    GGMLCGraph gf,
    GGMLCGraph gb,
    GGMLCGraph gb_tmp,
    GGMLTensor checkpoints,
    int n_checkpoints,
  ) {
    gg.ggml_build_backward_gradient_checkpointing(
      ptr,
      gf.ptr,
      gb.ptr,
      gb_tmp.ptr,
      ffi.Pointer.fromAddress(checkpoints.ptr.address),
      n_checkpoints,
    );
  }

  // optimize the function defined by the tensor f
  int opt(GGMLOptParams params, GGMLTensor f) {
    return gg.ggml_opt(ptr, params.ref, f.ptr);
  }

  // initialize optimizer context
  void optInit(GGMLOptContext opt, GGMLOptParams params, int nx) {
    gg.ggml_opt_init(ptr, opt.ptr, params.ref, nx);
  }

  // continue optimizing the function defined by the tensor f
  int optResume(GGMLOptContext opt, GGMLTensor f) {
    return gg.ggml_opt_resume(ptr, opt.ptr, f.ptr);
  }

  // Utils
  // Create a buffer and allocate all the tensors in a ggml_context
  GGMLBackendBuffer allocCtxTensorsFromBuft(GGMLBackendBufferType buft) {
    final p = gg.ggml_backend_alloc_ctx_tensors_from_buft(ptr, buft.ptr);
    return GGMLBackendBuffer.fromPtr(p);
  }

  GGMLBackendBuffer allocCtxTensors(GGMLBackend backend) {
    final p = gg.ggml_backend_alloc_ctx_tensors(ptr, backend.ptr);
    return GGMLBackendBuffer.fromPtr(p);
  }

  // TODO
  // int opt_resume_g(
  //   GgmlOptContext opt,
  //   GgmlTensor f,
  //   GgmlCGraph gf,
  //   GgmlCGraph gb,
  //   ggml_opt_callback callback,
  //   callback_data,
  // ) {}

  static final finalizer = ggFinalizer<ffi.Pointer<gg.ggml_context>>(ffi.Native.addressOf(gg.ggml_free));
}

class GGMLTensor extends GgStruct<gg.ggml_tensor> {
  GGMLTensor._(super.ptr) : super.fromPtr() {
    // finalizer.attach(this, ptr.cast());
  }

  factory GGMLTensor(GGMLContext ctx, int type, List<int> ne) {
    final pne = calloc<ffi.Int64>(ne.length);
    for (int i = 0; i < ne.length; i++) {
      pne[i] = ne[i];
    }
    final p = gg.ggml_new_tensor(ctx.ptr, type, ne.length, pne);
    return GGMLTensor._(p);
  }

  factory GGMLTensor.new1D(GGMLContext ctx, int type, int ne0) {
    final p = gg.ggml_new_tensor_1d(ctx.ptr, type, ne0);
    return GGMLTensor._(p);
  }

  factory GGMLTensor.new2D(GGMLContext ctx, int type, int ne0, int ne1) {
    final p = gg.ggml_new_tensor_2d(ctx.ptr, type, ne0, ne1);
    return GGMLTensor._(p);
  }

  factory GGMLTensor.new3D(GGMLContext ctx, int type, int ne0, int ne1, int ne2) {
    final p = gg.ggml_new_tensor_3d(ctx.ptr, type, ne0, ne1, ne2);
    return GGMLTensor._(p);
  }

  factory GGMLTensor.new4D(GGMLContext ctx, int type, int ne0, int ne1, int ne2, int ne3) {
    final p = gg.ggml_new_tensor_4d(ctx.ptr, type, ne0, ne1, ne2, ne3);
    return GGMLTensor._(p);
  }

  factory GGMLTensor.newI32(GGMLContext ctx, int value) {
    final p = gg.ggml_new_i32(ctx.ptr, value);
    return GGMLTensor._(p);
  }

  factory GGMLTensor.newF32(GGMLContext ctx, double value) {
    final p = gg.ggml_new_f32(ctx.ptr, value);
    return GGMLTensor._(p);
  }

  GGMLTensor setZero() {
    final p = gg.ggml_set_zero(ptr);
    return GGMLTensor._(p);
  }

  GGMLTensor setI32(int value) {
    final p = gg.ggml_set_i32(ptr, value);
    return GGMLTensor._(p);
  }

  GGMLTensor setF32(double value) {
    final p = gg.ggml_set_f32(ptr, value);
    return GGMLTensor._(p);
  }

  (int i0, int i1, int i2, int i3) unravelIndex(int i) {
    return using<(int, int, int, int)>((arena) {
      final pi0 = arena<ffi.Int64>();
      final pi1 = arena<ffi.Int64>();
      final pi2 = arena<ffi.Int64>();
      final pi3 = arena<ffi.Int64>();
      gg.ggml_unravel_index(ptr, i, pi0, pi1, pi2, pi3);
      return (pi0.value, pi1.value, pi2.value, pi3.value);
    });
  }

  int getI32_1D(int i) => gg.ggml_get_i32_1d(ptr, i);
  void setI32_1D(int i, int value) => gg.ggml_set_i32_1d(ptr, i, value);

  int getI32_ND(int i0, int i1, int i2, int i3) => gg.ggml_get_i32_nd(ptr, i0, i1, i2, i3);
  void setI32_ND(int i0, int i1, int i2, int i3, int value) => gg.ggml_set_i32_nd(ptr, i0, i1, i2, i3, value);

  double getF32_1D(int i) => gg.ggml_get_f32_1d(ptr, i);
  void setF32_1D(int i, double value) => gg.ggml_set_f32_1d(ptr, i, value);

  double getF32_ND(int i0, int i1, int i2, int i3) => gg.ggml_get_f32_nd(ptr, i0, i1, i2, i3);
  void setF32_ND(int i0, int i1, int i2, int i3, double value) =>
      gg.ggml_set_f32_nd(ptr, i0, i1, i2, i3, value);

  List<int> get ne => List.generate(4, (i) => ref.ne[i]);
  ffi.Pointer<ffi.Void> get data => gg.ggml_get_data(ptr);
  ffi.Pointer<ffi.Float> get dataF32 => gg.ggml_get_data_f32(ptr);

  int get unaryOp => gg.ggml_get_unary_op(ptr);
  String get name {
    return using<String>((arena) {
      final pname = gg.ggml_get_name(ptr);
      final name = pname.cast<Utf8>().toDartString();
      return name;
    });
  }

  set name(String n) {
    using<void>((arena) {
      final cname = n.toNativeUtf8(allocator: arena);
      gg.ggml_set_name(ptr, cname.cast());
    });
  }

  // struct ggml_tensor * ggml_format_name(      struct ggml_tensor * tensor, const char * fmt, ...);

  // static final finalizer = ffi.NativeFinalizer(gg.ggmltensor);

  @override
  gg.ggml_tensor get ref => ptr.ref;
}

class GGMLOptParams extends GgStruct<gg.ggml_opt_params> {
  GGMLOptParams._(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory GGMLOptParams(int type) {
    final s = gg.ggml_opt_default_params(type);
    final p = calloc<gg.ggml_opt_params>()..ref = s;
    return GGMLOptParams._(p);
  }

  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);
  @override
  gg.ggml_opt_params get ref => ptr.ref;
}

class GGMLOptContext extends GgStruct<gg.ggml_opt_context> {
  GGMLOptContext._(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory GGMLOptContext(GGMLContext ctx, GGMLOptParams params, int nx) {
    final p = calloc<gg.ggml_opt_context>();
    gg.ggml_opt_init(ctx.ptr, p, params.ref, nx);
    return GGMLOptContext._(p);
  }

  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);
  @override
  gg.ggml_opt_context get ref => ptr.ref;
}

class GGUFContext extends GgObject<gg.gguf_context> {
  GGUFContext.fromPtr(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory GGUFContext.empty() {
    final p = gg.gguf_init_empty();
    return GGUFContext.fromPtr(p);
  }

  factory GGUFContext.fromFile(String path, GGUFInitParams params) {
    final cpath = path.toNativeUtf8(allocator: calloc);
    final ctx = gg.gguf_init_from_file(cpath.cast(), params.ref);
    return GGUFContext.fromPtr(ctx);
  }

  int get version => gg.gguf_get_version(ptr);
  int get alignment => gg.gguf_get_alignment(ptr);
  int get dataOffset => gg.gguf_get_data_offset(ptr);
  ffi.Pointer<ffi.Void> get data => gg.gguf_get_data(ptr);

  int get nKV => gg.gguf_get_n_kv(ptr);

  int findKey(String name) {
    return using<int>((arena) {
      final cname = name.toNativeUtf8(allocator: arena);
      return gg.gguf_find_key(ptr, cname.cast());
    });
  }

  String getKey(int keyId) => gg.gguf_get_key(ptr, keyId).cast<Utf8>().toDartString();
  int getKvType(int keyId) => gg.gguf_get_kv_type(ptr, keyId);
  int getArrType(int keyId) => gg.gguf_get_arr_type(ptr, keyId);

  int get_val_u8(int keyId) => gg.gguf_get_val_u8(ptr, keyId);
  int get_val_i8(int keyId) => gg.gguf_get_val_i8(ptr, keyId);
  int get_val_u16(int keyId) => gg.gguf_get_val_u16(ptr, keyId);
  int get_val_i16(int keyId) => gg.gguf_get_val_i16(ptr, keyId);
  int get_val_u32(int keyId) => gg.gguf_get_val_u32(ptr, keyId);
  int get_val_i32(int keyId) => gg.gguf_get_val_i32(ptr, keyId);
  double get_val_f32(int keyId) => gg.gguf_get_val_f32(ptr, keyId);
  int get_val_u64(int keyId) => gg.gguf_get_val_u64(ptr, keyId);
  int get_val_i64(int keyId) => gg.gguf_get_val_i64(ptr, keyId);
  double get_val_f64(int keyId) => gg.gguf_get_val_f64(ptr, keyId);
  bool get_val_bool(int keyId) => gg.gguf_get_val_bool(ptr, keyId);
  String get_val_str(int keyId) => gg.gguf_get_val_str(ptr, keyId).cast<Utf8>().toDartString();
  ffi.Pointer<ffi.Void> get_val_data(int keyId) => gg.gguf_get_val_data(ptr, keyId);
  int get_arr_n(int keyId) => gg.gguf_get_arr_n(ptr, keyId);
  ffi.Pointer<ffi.Void> get_arr_data(int keyId) => gg.gguf_get_arr_data(ptr, keyId);
  String get_arr_str(int keyId, int i) => gg.gguf_get_arr_str(ptr, keyId, i).cast<Utf8>().toDartString();

  int get_n_tensors() => gg.gguf_get_n_tensors(ptr);
  int find_tensor(String name) {
    return using<int>((arena) {
      final cname = name.toNativeUtf8(allocator: arena);
      return gg.gguf_find_tensor(ptr, cname.cast());
    });
  }

  int get_tensor_offset(int i) => gg.gguf_get_tensor_offset(ptr, i);
  String get_tensor_name(int i) => gg.gguf_get_tensor_name(ptr, i).cast<Utf8>().toDartString();
  int get_tensor_type(int i) => gg.gguf_get_tensor_type(ptr, i);

  void set_val_u8(String key, int val) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_val_u8(ptr, ckey.cast(), val);
    });
  }

  void set_val_i8(String key, int val) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_val_i8(ptr, ckey.cast(), val);
    });
  }

  void set_val_u16(String key, int val) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_val_u16(ptr, ckey.cast(), val);
    });
  }

  void set_val_i16(String key, int val) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_val_i16(ptr, ckey.cast(), val);
    });
  }

  void set_val_u32(String key, int val) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_val_u32(ptr, ckey.cast(), val);
    });
  }

  void set_val_i32(String key, int val) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_val_i32(ptr, ckey.cast(), val);
    });
  }

  void set_val_f32(String key, double val) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_val_f32(ptr, ckey.cast(), val);
    });
  }

  void set_val_u64(String key, int val) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_val_u64(ptr, ckey.cast(), val);
    });
  }

  void set_val_i64(String key, int val) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_val_i64(ptr, ckey.cast(), val);
    });
  }

  void set_val_f64(String key, double val) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_val_f64(ptr, ckey.cast(), val);
    });
  }

  void set_val_bool(String key, bool val) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_val_bool(ptr, ckey.cast(), val);
    });
  }

  void set_val_str(String key, String val) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      final cval = val.toNativeUtf8(allocator: arena);
      gg.gguf_set_val_str(ptr, ckey.cast(), cval.cast());
    });
  }

  void set_arr_data(String key, int type, ffi.Pointer<ffi.Void> data, int n) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_arr_data(ptr, ckey.cast(), type, data, n);
    });
  }

  void set_arr_str(String key, ffi.Pointer<ffi.Pointer<ffi.Char>> data, int n) {
    using<void>((arena) {
      final ckey = key.toNativeUtf8(allocator: arena);
      gg.gguf_set_arr_str(ptr, ckey.cast(), data, n);
    });
  }

  // set or add KV pairs from another context
  void set_kv(GGUFContext src) => gg.gguf_set_kv(ptr, src.ptr);

  // manage tensor info
  void add_tensor(GGMLTensor tensor) => gg.gguf_add_tensor(ptr, tensor.ptr);
  void set_tensor_type(String name, int type) =>
      using((arena) => gg.gguf_set_tensor_type(ptr, name.toNativeUtf8(allocator: arena).cast(), type));
  void set_tensor_data(String name, ffi.Pointer<ffi.Void> data, int size) =>
      using((arena) => gg.gguf_set_tensor_data(ptr, name.toNativeUtf8(allocator: arena).cast(), data, size));

  // writing gguf files can be done in 2 ways:
  //
  // - write the entire gguf_context to a binary file in a single pass:
  //
  //   gguf_write_to_file(ctx, fname);
  //
  // - first prepare a file with a placeholder for the meta data, write the tensor data, then write the meta data:
  //
  //   FILE * f = fopen(fname, "wb");
  //   fseek(f, gguf_get_meta_size(ctx), SEEK_SET);
  //   fwrite(f, ...);
  //   void * data = gguf_meta_get_meta_data(ctx);
  //   fseek(f, 0, SEEK_SET);
  //   fwrite(f, data, gguf_get_meta_size(ctx));
  //   free(data);
  //   fclose(f);
  //

  // write the entire context to a binary file
  void WriteToFile(String fname, bool onlyMeta) {
    using<void>((arena) {
      final cname = fname.toNativeUtf8(allocator: arena);
      gg.gguf_write_to_file(ptr, cname.cast(), onlyMeta);
    });
  }

  // get the size in bytes of the meta data (header, kv pairs, tensor info) including padding
  int get_meta_size() => gg.gguf_get_meta_size(ptr);

  // ffi.Pointer<ffi.Void> get_meta_data() {
  //   final p = calloc<ffi.Void>();
  //   gg.gguf_get_meta_data(ptr, p);
  //   return p;
  // }

  // factory GgufContext.fromBuffer(Uint8List data, GgufInitParams params){}

  static String typeName(int type) => gg.gguf_type_name(type).cast<Utf8>().toDartString();

  static final finalizer = ggFinalizer<ffi.Pointer<gg.gguf_context>>(ffi.Native.addressOf(gg.gguf_free));
}

class GGUFInitParams extends GgStruct<gg.gguf_init_params> {
  GGUFInitParams._(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory GGUFInitParams() {
    final p = calloc<gg.gguf_init_params>();
    return GGUFInitParams._(p);
  }

  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);
}
