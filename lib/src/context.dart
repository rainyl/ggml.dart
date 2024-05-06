import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'backend.dart';
import 'base.dart';
import 'graph.dart';
import 'params.dart';
import 'tensor.dart';
import 'ggml.g.dart' as gg;

class Context extends GGBase<gg.ggml_context> {
  Context._(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory Context.init(InitParams params) {
    final p = gg.ggml_init(params.ref);
    return Context._(p);
  }

  Tensor newTensor(int type, List<int> ne) {
    final pne = calloc<ffi.Int64>(ne.length);
    for (int i = 0; i < ne.length; i++) {
      pne[i] = ne[i];
    }
    final p = gg.ggml_new_tensor(ptr, type, ne.length, pne);
    return Tensor.fromPtr(p);
  }

  Tensor newTensor1D(int type, int ne0) {
    final p = gg.ggml_new_tensor_1d(ptr, type, ne0);
    return Tensor.fromPtr(p);
  }

  Tensor newTensor2D(int type, int ne0, int ne1) {
    final p = gg.ggml_new_tensor_2d(ptr, type, ne0, ne1);
    return Tensor.fromPtr(p);
  }

  Tensor newTensor3D(int type, int ne0, int ne1, int ne2) {
    final p = gg.ggml_new_tensor_3d(ptr, type, ne0, ne1, ne2);
    return Tensor.fromPtr(p);
  }

  Tensor newTensor4D(int type, int ne0, int ne1, int ne2, int ne3) {
    final p = gg.ggml_new_tensor_4d(ptr, type, ne0, ne1, ne2, ne3);
    return Tensor.fromPtr(p);
  }

  Tensor dupTensor(Tensor src) {
    final p = gg.ggml_dup_tensor(ptr, src.ptr);
    return Tensor.fromPtr(p);
  }

  Tensor viewTensor(Tensor src) {
    final p = gg.ggml_view_tensor(ptr, src.ptr);
    return Tensor.fromPtr(p);
  }

  Tensor getFirstTensor() {
    final p = gg.ggml_get_first_tensor(ptr);
    return Tensor.fromPtr(p);
  }

  Tensor getNextTensor(Tensor src) {
    final p = gg.ggml_get_next_tensor(ptr, src.ptr);
    return Tensor.fromPtr(p);
  }

  Tensor getTensor(String name) {
    return using<Tensor>((arena) {
      final cname = name.toNativeUtf8(allocator: arena);
      final p = gg.ggml_get_next_tensor(ptr, cname.cast());
      return Tensor.fromPtr(p);
    });
  }

  Tensor dup(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_dup_inplace(ptr, a.ptr) : gg.ggml_dup(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  Tensor add(Tensor a, Tensor b, [bool inplace = false, bool cast = false, int? castType]) {
    if (cast && castType != null) {
      return Tensor.fromPtr(gg.ggml_add_cast(ptr, a.ptr, b.ptr, castType));
    }
    final p = inplace ? gg.ggml_add_inplace(ptr, a.ptr, b.ptr) : gg.ggml_add(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  Tensor add1(Tensor a, Tensor b, [bool inplace = false]) {
    final p = inplace ? gg.ggml_add1_inplace(ptr, a.ptr, b.ptr) : gg.ggml_add1(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  Tensor acc(Tensor a, Tensor b, int nb1, int nb2, int nb3, int offset, [bool inplace = false]) {
    final p = inplace
        ? gg.ggml_acc_inplace(ptr, a.ptr, b.ptr, nb1, nb2, nb3, offset)
        : gg.ggml_acc(ptr, a.ptr, b.ptr, nb1, nb2, nb3, offset);
    return Tensor.fromPtr(p);
  }

  Tensor sub(Tensor a, Tensor b, [bool inplace = false]) {
    final p = inplace ? gg.ggml_sub_inplace(ptr, a.ptr, b.ptr) : gg.ggml_sub(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  Tensor mul(Tensor a, Tensor b, [bool inplace = false]) {
    final p = inplace ? gg.ggml_mul_inplace(ptr, a.ptr, b.ptr) : gg.ggml_mul(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  Tensor div(Tensor a, Tensor b, [bool inplace = false]) {
    final p = inplace ? gg.ggml_div_inplace(ptr, a.ptr, b.ptr) : gg.ggml_div(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  Tensor sqr(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_sqr_inplace(ptr, a.ptr) : gg.ggml_sqr(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  Tensor sqrt(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_sqrt_inplace(ptr, a.ptr) : gg.ggml_sqrt(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  Tensor log(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_log_inplace(ptr, a.ptr) : gg.ggml_log(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  /// sum
  /// return scalar
  Tensor sum(Tensor a) {
    final p = gg.ggml_sum(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  /// sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
  Tensor sumRows(Tensor a) {
    final p = gg.ggml_sum_rows(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  /// mean along rows
  Tensor mean(Tensor a) {
    final p = gg.ggml_mean(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  /// argmax along rows
  Tensor argmax(Tensor a) {
    final p = gg.ggml_argmax(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // if a is the same shape as b, and a is not parameter, return a
  // otherwise, return a new tensor: repeat(a) to fit in b
  Tensor repeat(Tensor a, Tensor b) {
    final p = gg.ggml_repeat(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  // sums repetitions in a into shape of b
  Tensor repeatBack(Tensor a, Tensor b) {
    final p = gg.ggml_repeat_back(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  // concat a and b on dim 2
  // used in stable-diffusion
  Tensor concat(Tensor a, Tensor b) {
    final p = gg.ggml_concat(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  // abs
  Tensor abs(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_abs_inplace(ptr, a.ptr) : gg.ggml_abs(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // sgn
  Tensor sgn(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_sgn_inplace(ptr, a.ptr) : gg.ggml_sgn(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // neg
  Tensor neg(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_neg_inplace(ptr, a.ptr) : gg.ggml_neg(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  //tanh
  Tensor tanh(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_tanh_inplace(ptr, a.ptr) : gg.ggml_tanh(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // elu
  Tensor elu(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_elu_inplace(ptr, a.ptr) : gg.ggml_elu(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // relu
  Tensor relu(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_relu_inplace(ptr, a.ptr) : gg.ggml_relu(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // leaky_relu
  Tensor leakyRelu(Tensor a, double negativeSlope, [bool inplace = false]) {
    final p = gg.ggml_leaky_relu(ptr, a.ptr, negativeSlope, inplace);
    return Tensor.fromPtr(p);
  }

  // gelu
  Tensor gelu(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_gelu_inplace(ptr, a.ptr) : gg.ggml_gelu(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // gelu_quick
  Tensor geluQuick(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_gelu_quick_inplace(ptr, a.ptr) : gg.ggml_gelu_quick(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // silu
  Tensor silu(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_silu_inplace(ptr, a.ptr) : gg.ggml_silu(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // a - x
  // b - dy
  Tensor siluBack(Tensor a, Tensor b) {
    final p = gg.ggml_silu_back(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  // hardswish(x) = x * relu6(x + 3) / 6
  Tensor hardswish(Tensor a) {
    final p = gg.ggml_hardswish(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // hardsigmoid(x) = relu6(x + 3) / 6
  Tensor hardsigmoid(Tensor a) {
    final p = gg.ggml_hardsigmoid(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // normalize along rows
  Tensor norm(Tensor a, double eps, [bool inplace = false]) {
    final p = inplace ? gg.ggml_norm_inplace(ptr, a.ptr, eps) : gg.ggml_norm(ptr, a.ptr, eps);
    return Tensor.fromPtr(p);
  }

  // rms_norm
  Tensor rmsNorm(Tensor a, double eps, [bool inplace = false]) {
    final p = inplace ? gg.ggml_rms_norm_inplace(ptr, a.ptr, eps) : gg.ggml_rms_norm(ptr, a.ptr, eps);
    return Tensor.fromPtr(p);
  }

  // group normalize along ne0*ne1*n_groups
  // used in stable-diffusion
  // TODO: eps is hardcoded to 1e-6 for now
  Tensor groupNorm(Tensor a, int nGroups, [bool inplace = false]) {
    final p =
        inplace ? gg.ggml_group_norm_inplace(ptr, a.ptr, nGroups) : gg.ggml_group_norm(ptr, a.ptr, nGroups);
    return Tensor.fromPtr(p);
  }

  // a - x
  // b - dy
  Tensor rmsNormBack(Tensor a, Tensor b, double eps) {
    final p = gg.ggml_rms_norm_back(ptr, a.ptr, b.ptr, eps);
    return Tensor.fromPtr(p);
  }

  // A: k columns, n rows => [ne03, ne02, n, k]
  // B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
  // result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
  Tensor mulMat(Tensor a, Tensor b) {
    final p = gg.ggml_mul_mat(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  // change the precision of a matrix multiplication
  // set to GGML_PREC_F32 for higher precision (useful for phi-2)
  void mulMatSetPrec(Tensor a, int prec) {
    gg.ggml_mul_mat_set_prec(a.ptr, prec);
  }

  // indirect matrix multiplication
  //  ggml_mul_mat_id(ctx, as, ids, id, b) ~= ggml_mul_mat(as[ids[id]], b)
  Tensor mulMatId(Tensor as, Tensor ids, int id, Tensor b) {
    final p = gg.ggml_mul_mat_id(ptr, as.ptr, ids.ptr, id, b.ptr);
    return Tensor.fromPtr(p);
  }

  // A: m columns, n rows,
  // B: p columns, n rows,
  // result is m columns, p rows
  Tensor outProd(Tensor a, Tensor b) {
    final p = gg.ggml_out_prod(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  //
  // operations on tensors without backpropagation
  //

  Tensor scale(Tensor a, double s, [bool inplace = false]) {
    final p = inplace ? gg.ggml_scale_inplace(ptr, a.ptr, s) : gg.ggml_scale(ptr, a.ptr, s);
    return Tensor.fromPtr(p);
  }

  // set
  Tensor set(Tensor a, Tensor b, int nb1, int nb2, int nb3, int offset, [bool inplace = false]) {
    final p = inplace
        ? gg.ggml_set_inplace(ptr, a.ptr, b.ptr, nb1, nb2, nb3, offset)
        : gg.ggml_set(ptr, a.ptr, b.ptr, nb1, nb2, nb3, offset);
    return Tensor.fromPtr(p);
  }

  Tensor set1D(Tensor a, Tensor b, int offset, [bool inplace = false]) {
    final p = inplace
        ? gg.ggml_set_1d_inplace(ptr, a.ptr, b.ptr, offset)
        : gg.ggml_set_1d(ptr, a.ptr, b.ptr, offset);
    return Tensor.fromPtr(p);
  }

  /// b -> view(a,offset,nb1,nb2,3)
  Tensor set2D(Tensor a, Tensor b, int nb1, int offset, [bool inplace = false]) {
    final p = inplace
        ? gg.ggml_set_2d_inplace(ptr, a.ptr, b.ptr, nb1, offset)
        : gg.ggml_set_2d(ptr, a.ptr, b.ptr, nb1, offset);
    return Tensor.fromPtr(p);
  }

  /// a -> b, return view(b)
  Tensor cpy(Tensor a, Tensor b) {
    final p = gg.ggml_cpy(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  // cast
  Tensor cast(Tensor a, int type) {
    final p = gg.ggml_cast(ptr, a.ptr, type);
    return Tensor.fromPtr(p);
  }

  // make contiguous
  Tensor cont(Tensor a) {
    final p = gg.ggml_cont(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // make contiguous, with new shape
  Tensor cont1D(Tensor a, int ne0) {
    final p = gg.ggml_cont_1d(ptr, a.ptr, ne0);
    return Tensor.fromPtr(p);
  }

  Tensor cont2D(Tensor a, int ne0, int ne1) {
    final p = gg.ggml_cont_2d(ptr, a.ptr, ne0, ne1);
    return Tensor.fromPtr(p);
  }

  Tensor cont3D(Tensor a, int ne0, int ne1, int ne2) {
    final p = gg.ggml_cont_3d(ptr, a.ptr, ne0, ne1, ne2);
    return Tensor.fromPtr(p);
  }

  Tensor cont4D(Tensor a, int ne0, int ne1, int ne2, int ne3) {
    final p = gg.ggml_cont_4d(ptr, a.ptr, ne0, ne1, ne2, ne3);
    return Tensor.fromPtr(p);
  }

  // return view(a), b specifies the new shape
  // TODO: when we start computing gradient, make a copy instead of view
  Tensor reshape(Tensor a, Tensor b) {
    final p = gg.ggml_reshape(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  // return view(a)
  // TODO: when we start computing gradient, make a copy instead of view
  Tensor reshape1D(Tensor a, int ne0) {
    final p = gg.ggml_reshape_1d(ptr, a.ptr, ne0);
    return Tensor.fromPtr(p);
  }

  Tensor reshape2D(Tensor a, int ne0, int ne1) {
    final p = gg.ggml_reshape_2d(ptr, a.ptr, ne0, ne1);
    return Tensor.fromPtr(p);
  }

  Tensor reshape3D(Tensor a, int ne0, int ne1, int ne2) {
    final p = gg.ggml_reshape_3d(ptr, a.ptr, ne0, ne1, ne2);
    return Tensor.fromPtr(p);
  }

  Tensor reshape4D(Tensor a, int ne0, int ne1, int ne2, int ne3) {
    final p = gg.ggml_reshape_4d(ptr, a.ptr, ne0, ne1, ne2, ne3);
    return Tensor.fromPtr(p);
  }

  // offset in bytes
  Tensor view1D(Tensor a, int ne0, int offset) {
    final p = gg.ggml_view_1d(ptr, a.ptr, ne0, offset);
    return Tensor.fromPtr(p);
  }

  Tensor view2D(Tensor a, int ne0, int ne1, int nb1, int offset) {
    final p = gg.ggml_view_2d(ptr, a.ptr, ne0, ne1, nb1, offset);
    return Tensor.fromPtr(p);
  }

  Tensor view3D(Tensor a, int ne0, int ne1, int ne2, int nb1, int nb2, int offset) {
    final p = gg.ggml_view_3d(ptr, a.ptr, ne0, ne1, ne2, nb1, nb2, offset);
    return Tensor.fromPtr(p);
  }

  Tensor view4D(Tensor a, int ne0, int ne1, int ne2, int ne3, int nb1, int nb2, int nb3, int offset) {
    final p = gg.ggml_view_4d(ptr, a.ptr, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);
    return Tensor.fromPtr(p);
  }

  Tensor permute(Tensor a, int axis0, int axis1, int axis2, int axis3) {
    final p = gg.ggml_permute(ptr, a.ptr, axis0, axis1, axis2, axis3);
    return Tensor.fromPtr(p);
  }

  // alias for ggml_permute(ctx, a, 1, 0, 2, 3)
  Tensor transpose(Tensor a) {
    final p = gg.ggml_transpose(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // supports 3D: a->ne[2] == b->ne[1]
  Tensor getRows(Tensor a, Tensor b) {
    final p = gg.ggml_get_rows(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  Tensor getRowsBack(Tensor a, Tensor b, Tensor c) {
    final p = gg.ggml_get_rows_back(ptr, a.ptr, b.ptr, c.ptr);
    return Tensor.fromPtr(p);
  }

  // diag
  Tensor diag(Tensor a) {
    final p = gg.ggml_diag(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // set elements above the diagonal to -INF
  Tensor diagMaskInf(Tensor a, int n, [bool inplace = false]) {
    final p = inplace ? gg.ggml_diag_mask_inf_inplace(ptr, a.ptr, n) : gg.ggml_diag_mask_inf(ptr, a.ptr, n);
    return Tensor.fromPtr(p);
  }

  // set elements above the diagonal to 0
  Tensor diagMaskZero(Tensor a, int n, [bool inplace = false]) {
    final p = inplace ? gg.ggml_diag_mask_zero_inplace(ptr, a.ptr, n) : gg.ggml_diag_mask_zero(ptr, a.ptr, n);
    return Tensor.fromPtr(p);
  }

  // softmax
  Tensor softMax(Tensor a, [bool inplace = false]) {
    final p = inplace ? gg.ggml_soft_max_inplace(ptr, a.ptr) : gg.ggml_soft_max(ptr, a.ptr);
    return Tensor.fromPtr(p);
  }

  // fused soft_max(a*scale + mask + pos[i]*(ALiBi slope))
  // mask is optional
  // pos is required when max_bias > 0.0f
  // max_bias = 0.0f for no ALiBi
  Tensor softMaxExt(Tensor a, Tensor mask, Tensor pos, double scale, double maxBias) {
    final p = gg.ggml_soft_max_ext(ptr, a.ptr, mask.ptr, pos.ptr, scale, maxBias);
    return Tensor.fromPtr(p);
  }

  Tensor softMaxBack(Tensor a, Tensor b, [bool inplace = false]) {
    final p =
        inplace ? gg.ggml_soft_max_back_inplace(ptr, a.ptr, b.ptr) : gg.ggml_soft_max_back(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  // rotary position embedding
  // if mode & 1 == 1, skip n_past elements (DEPRECATED)
  // if mode & 2 == 1, GPT-NeoX style
  // if mode & 4 == 1, ChatGLM style
  //
  // b is an int32 vector with size a->ne[2], it contains the positions
  Tensor rope(Tensor a, Tensor b, int nDims, int mode, int nCtx, [bool inplace = false]) {
    final p = inplace
        ? gg.ggml_rope_inplace(ptr, a.ptr, b.ptr, nDims, mode, nCtx)
        : gg.ggml_rope(ptr, a.ptr, b.ptr, nDims, mode, nCtx);
    return Tensor.fromPtr(p);
  }

  // custom RoPE
  Tensor ropeCustom(
    Tensor a,
    Tensor b,
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
    return Tensor.fromPtr(p);
  }

  /// compute correction dims for YaRN RoPE scaling
  (double, double) ropeYarnCorrDims(
      int nDims, int nOrigCtx, double freqBase, double betaFast, double betaSlow) {
    return using<(double, double)>((arena) {
      final pres = arena<ffi.Float>(2);
      gg.ggml_rope_yarn_corr_dims(nDims, nOrigCtx, freqBase, betaFast, betaSlow, pres);
      return (pres[0], pres[1]);
    });
  }

  // xPos RoPE, in-place, returns view(a)
  Tensor ropeXposInplace(Tensor a, Tensor b, int nDims, double base, bool down) {
    final p = gg.ggml_rope_xpos_inplace(ptr, a.ptr, b.ptr, nDims, base, down);
    return Tensor.fromPtr(p);
  }

  // rotary position embedding backward, i.e compute dx from dy
  // a - dy
  Tensor ropeBack(
      Tensor a,
      Tensor b,
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
    return Tensor.fromPtr(p);
  }

  // clamp
  Tensor clamp(Tensor a, double min, double max) {
    final p = gg.ggml_clamp(ptr, a.ptr, min, max);
    return Tensor.fromPtr(p);
  }

  // im2col
  Tensor im2col(Tensor a, Tensor b, int s0, int s1, int p0, int p1, int d0, int d1, bool is2D, int dstType) {
    final p = gg.ggml_im2col(ptr, a.ptr, b.ptr, s0, s1, p0, p1, d0, d1, is2D, dstType);
    return Tensor.fromPtr(p);
  }

  // conv_depthwise_2d
  Tensor convDepthwise2d(Tensor a, Tensor b, int s0, int s1, int p0, int p1, int d0, int d1) {
    final p = gg.ggml_conv_depthwise_2d(ptr, a.ptr, b.ptr, s0, s1, p0, p1, d0, d1);
    return Tensor.fromPtr(p);
  }

  // conv_1d
  Tensor conv1d(Tensor a, Tensor b, int s0, int p0, int d0) {
    final p = gg.ggml_conv_1d(ptr, a.ptr, b.ptr, s0, p0, d0);
    return Tensor.fromPtr(p);
  }

  // conv_1d with padding = half
  // alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)
  Tensor conv1dPh(Tensor a, Tensor b, int s, int d) {
    final p = gg.ggml_conv_1d_ph(ptr, a.ptr, b.ptr, s, d);
    return Tensor.fromPtr(p);
  }

  // conv_transpose_1d
  Tensor convTranspose1d(Tensor a, Tensor b, int s0, int p0, int d0) {
    final p = gg.ggml_conv_transpose_1d(ptr, a.ptr, b.ptr, s0, p0, d0);
    return Tensor.fromPtr(p);
  }

  // conv_2d
  Tensor conv2d(Tensor a, Tensor b, int s0, int s1, int p0, int p1, int d0, int d1) {
    final p = gg.ggml_conv_2d(ptr, a.ptr, b.ptr, s0, s1, p0, p1, d0, d1);
    return Tensor.fromPtr(p);
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
  Tensor conv2dSkP0(Tensor a, Tensor b) {
    final p = gg.ggml_conv_2d_sk_p0(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
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
  Tensor conv2dS1Ph(Tensor a, Tensor b) {
    final p = gg.ggml_conv_2d_s1_ph(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  // conv_transpose_2d_p0
  Tensor convTranspose2dP0(Tensor a, Tensor b, int stride) {
    final p = gg.ggml_conv_transpose_2d_p0(ptr, a.ptr, b.ptr, stride);
    return Tensor.fromPtr(p);
  }

  // pool_1d
  Tensor pool1d(Tensor a, int op, int k0, int s0, int p0) {
    final p = gg.ggml_pool_1d(ptr, a.ptr, op, k0, s0, p0);
    return Tensor.fromPtr(p);
  }

  // the result will have 2*p0 padding for the first dimension
  // and 2*p1 padding for the second dimension
  Tensor pool2d(Tensor a, int op, int k0, int k1, int s0, int s1, double p0, double p1) {
    final p = gg.ggml_pool_2d(ptr, a.ptr, op, k0, k1, s0, s1, p0, p1);
    return Tensor.fromPtr(p);
  }

  // nearest interpolate
  // used in stable-diffusion
  // upscale
  Tensor upscale(Tensor a, int scaleFactor) {
    final p = gg.ggml_upscale(ptr, a.ptr, scaleFactor);
    return Tensor.fromPtr(p);
  }

  // pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]
  Tensor pad(Tensor a, int p0, int p1, int p2, int p3) {
    final p = gg.ggml_pad(ptr, a.ptr, p0, p1, p2, p3);
    return Tensor.fromPtr(p);
  }

  // Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
  // timesteps: [N,]
  // return: [N, dim]
  Tensor timestepEmbedding(Tensor timesteps, int dim, int maxPeriod) {
    final p = gg.ggml_timestep_embedding(ptr, timesteps.ptr, dim, maxPeriod);
    return Tensor.fromPtr(p);
  }

  // argsort
  Tensor argsort(Tensor a, int order) {
    final p = gg.ggml_argsort(ptr, a.ptr, order);
    return Tensor.fromPtr(p);
  }

  // arange
  Tensor arange(double start, double stop, double step) {
    final p = gg.ggml_arange(ptr, start, stop, step);
    return Tensor.fromPtr(p);
  }

  // top k elements per row
  Tensor topk(Tensor a, int k) {
    final p = gg.ggml_top_k(ptr, a.ptr, k);
    return Tensor.fromPtr(p);
  }

  // flash_attn
  Tensor flashAttn(Tensor q, Tensor k, Tensor v, bool masked) {
    final p = gg.ggml_flash_attn(ptr, q.ptr, k.ptr, v.ptr, masked);
    return Tensor.fromPtr(p);
  }

  // flash_attn_back
  Tensor flashAttnBack(Tensor q, Tensor k, Tensor v, Tensor d, bool masked) {
    final p = gg.ggml_flash_attn_back(ptr, q.ptr, k.ptr, v.ptr, d.ptr, masked);
    return Tensor.fromPtr(p);
  }

  // flash_ff
  Tensor flashFf(Tensor a, Tensor b0, Tensor b1, Tensor c0, Tensor c1) {
    final p = gg.ggml_flash_ff(ptr, a.ptr, b0.ptr, b1.ptr, c0.ptr, c1.ptr);
    return Tensor.fromPtr(p);
  }

  // ssm_conv
  Tensor ssmConv(Tensor s, Tensor x, Tensor c, Tensor sq) {
    final p = gg.ggml_ssm_conv(ptr, s.ptr, x.ptr, c.ptr, sq.ptr);
    return Tensor.fromPtr(p);
  }

  // ssm_scan
  Tensor ssmScan(Tensor s, Tensor x, Tensor dt, Tensor A, Tensor B, Tensor C, Tensor sq) {
    final p = gg.ggml_ssm_scan(ptr, s.ptr, x.ptr, dt.ptr, A.ptr, B.ptr, C.ptr, sq.ptr);
    return Tensor.fromPtr(p);
  }

  // partition into non-overlapping windows with padding if needed
  // example:
  // a:   768   64   64    1
  // w:    14
  // res: 768   14   14    25
  // used in sam
  Tensor winPart(Tensor a, int w) {
    final p = gg.ggml_win_part(ptr, a.ptr, w);
    return Tensor.fromPtr(p);
  }

  // reverse of ggml_win_part
  // used in sam
  Tensor winUnpart(Tensor a, int w0, int h0, int w) {
    final p = gg.ggml_win_unpart(ptr, a.ptr, w0, h0, w);
    return Tensor.fromPtr(p);
  }

  // unary
  Tensor unary(Tensor a, int op, [bool inplace = false]) {
    final p = inplace ? gg.ggml_unary_inplace(ptr, a.ptr, op) : gg.ggml_unary(ptr, a.ptr, op);
    return Tensor.fromPtr(p);
  }

  // get_rel_pos
  Tensor getRelPos(Tensor a, int qh, int kh) {
    final p = gg.ggml_get_rel_pos(ptr, a.ptr, qh, kh);
    return Tensor.fromPtr(p);
  }

  // add_rel_pos
  Tensor addRelPos(Tensor a, Tensor pw, Tensor ph, [bool inplace = false]) {
    final p = inplace
        ? gg.ggml_add_rel_pos_inplace(ptr, a.ptr, pw.ptr, ph.ptr)
        : gg.ggml_add_rel_pos(ptr, a.ptr, pw.ptr, ph.ptr);
    return Tensor.fromPtr(p);
  }

  // custom operators v2
  // map_custom1
  // GgmlTensor mapCustom1(GgmlTensor a, int op, int nTasks, ffi.Pointer<ffi.Void> userdata){
  //   final p = gg.ggml_map_custom1(ptr, a.ptr, op, nTasks, userdata);
  // }

  // loss function
  // cross_entropy_loss
  Tensor crossEntropyLoss(Tensor a, Tensor b) {
    final p = gg.ggml_cross_entropy_loss(ptr, a.ptr, b.ptr);
    return Tensor.fromPtr(p);
  }

  // cross_entropy_loss_back
  Tensor crossEntropyLossBack(Tensor a, Tensor b, Tensor c) {
    final p = gg.ggml_cross_entropy_loss_back(ptr, a.ptr, b.ptr, c.ptr);
    return Tensor.fromPtr(p);
  }

  //
  // automatic differentiation
  //
  // set_param
  void setParam(Tensor a) {
    gg.ggml_set_param(ptr, a.ptr);
  }

  // build_forward_expand
  void buildForwardExpand(CGraph cgraph, Tensor tensor) {
    gg.ggml_build_forward_expand(cgraph.ptr, tensor.ptr);
  }

  // build_backward_expand
  void buildBackwardExpand(CGraph gf, CGraph gb, bool keep) {
    gg.ggml_build_backward_expand(ptr, gf.ptr, gb.ptr, keep);
  }

  // graph allocation in a context
  CGraph newGraph() {
    final p = gg.ggml_new_graph(ptr);
    return CGraph.fromPtr(p);
  }

  CGraph newGraphCustom(int size, bool grads) {
    final p = gg.ggml_new_graph_custom(ptr, size, grads);
    return CGraph.fromPtr(p);
  }

  CGraph graphDup(CGraph cgraph) {
    final p = gg.ggml_graph_dup(ptr, cgraph.ptr);
    return CGraph.fromPtr(p);
  }

  // build gradient checkpointing backward graph gb for gf using provided checkpoints
  // gb_tmp will contain original backward graph with rewritten backward process nodes,
  // but without the second forward pass nodes.
  void buildBackwardGradientCheckpointing(
    CGraph gf,
    CGraph gb,
    CGraph gbTmp,
    Tensor checkpoints,
    int nCheckpoints,
  ) {
    gg.ggml_build_backward_gradient_checkpointing(
      ptr,
      gf.ptr,
      gb.ptr,
      gbTmp.ptr,
      ffi.Pointer.fromAddress(checkpoints.ptr.address),
      nCheckpoints,
    );
  }

  // optimize the function defined by the tensor f
  int opt(OptParams params, Tensor f) {
    return gg.ggml_opt(ptr, params.ref, f.ptr);
  }

  // initialize optimizer context
  void optInit(OptContext opt, OptParams params, int nx) {
    gg.ggml_opt_init(ptr, opt.ptr, params.ref, nx);
  }

  // continue optimizing the function defined by the tensor f
  int optResume(OptContext opt, Tensor f) {
    return gg.ggml_opt_resume(ptr, opt.ptr, f.ptr);
  }

  // Utils
  // Create a buffer and allocate all the tensors in a ggml_context
  BackendBuffer allocCtxTensorsFromBuft(BackendBufferType buft) {
    final p = gg.ggml_backend_alloc_ctx_tensors_from_buft(ptr, buft.ptr);
    return BackendBuffer.fromPtr(p);
  }

  BackendBuffer allocCtxTensors(Backend backend) {
    final p = gg.ggml_backend_alloc_ctx_tensors(ptr, backend.ptr);
    return BackendBuffer.fromPtr(p);
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

class OptContext extends GGStruct<gg.ggml_opt_context> {
  OptContext._(super.ptr) : super.fromPtr() {
    // finalizer.attach(this, ptr.cast());
  }

  factory OptContext(Context ctx, OptParams params, int nx) {
    final p = calloc<gg.ggml_opt_context>();
    gg.ggml_opt_init(ctx.ptr, p, params.ref, nx);
    return OptContext._(p);
  }

  // static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);
  @override
  gg.ggml_opt_context get ref => ptr.ref;
}

class GGUFContext extends GGBase<gg.gguf_context> {
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
  void add_tensor(Tensor tensor) => gg.gguf_add_tensor(ptr, tensor.ptr);
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
  void writeToFile(String fname, bool onlyMeta) {
    using<void>((arena) {
      final cname = fname.toNativeUtf8(allocator: arena);
      gg.gguf_write_to_file(ptr, cname.cast(), onlyMeta);
    });
  }

  // get the size in bytes of the meta data (header, kv pairs, tensor info) including padding
  int getMetaSize() => gg.gguf_get_meta_size(ptr);

  // ffi.Pointer<ffi.Void> get_meta_data() {
  //   final p = calloc<ffi.Void>();
  //   gg.gguf_get_meta_data(ptr, p);
  //   return p;
  // }

  // factory GgufContext.fromBuffer(Uint8List data, GgufInitParams params){}

  static String typeName(int type) => gg.gguf_type_name(type).cast<Utf8>().toDartString();

  static final finalizer = ggFinalizer<ffi.Pointer<gg.gguf_context>>(ffi.Native.addressOf(gg.gguf_free));
}

class GGUFInitParams extends GGStruct<gg.gguf_init_params> {
  GGUFInitParams._(super.ptr) : super.fromPtr() {
    // finalizer.attach(this, ptr.cast());
  }

  factory GGUFInitParams() {
    final p = calloc<gg.gguf_init_params>();
    return GGUFInitParams._(p);
  }

  // static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);
}
