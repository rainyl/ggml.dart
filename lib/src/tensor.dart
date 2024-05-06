// ignore_for_file: non_constant_identifier_names

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'constants.dart';
import 'context.dart';
import 'ggml.g.dart' as gg;

class Tensor extends GGStruct<gg.ggml_tensor> {
  Tensor.fromPtr(super.ptr) : super.fromPtr() {
    // finalizer.attach(this, ptr.cast());
  }

  factory Tensor(Context ctx, int type, List<int> ne) {
    final pne = calloc<ffi.Int64>(ne.length);
    for (int i = 0; i < ne.length; i++) {
      pne[i] = ne[i];
    }
    final p = gg.ggml_new_tensor(ctx.ptr, type, ne.length, pne);
    return Tensor.fromPtr(p);
  }

  factory Tensor.new1D(Context ctx, int type, int ne0, {List<num>? data}) {
    final res = Tensor.fromPtr(gg.ggml_new_tensor_1d(ctx.ptr, type, ne0));
    if (data != null) {
      res.setData(data);
    }
    return res;
  }
  factory Tensor.new2D(Context ctx, int type, int ne0, int ne1, {List<num>? data}) {
    final res = Tensor.fromPtr(gg.ggml_new_tensor_2d(ctx.ptr, type, ne0, ne1));
    if (data != null) {
      res.setData(data);
    }
    return res;
  }

  factory Tensor.new3D(Context ctx, int type, int ne0, int ne1, int ne2, {List<num>? data}) {
    final res = Tensor.fromPtr(gg.ggml_new_tensor_3d(ctx.ptr, type, ne0, ne1, ne2));
    if (data != null) {
      res.setData(data);
    }
    return res;
  }
  factory Tensor.new4D(Context ctx, int type, int ne0, int ne1, int ne2, int ne3, {List<num>? data}) {
    final res = Tensor.fromPtr(gg.ggml_new_tensor_4d(ctx.ptr, type, ne0, ne1, ne2, ne3));
    if (data != null) {
      res.setData(data);
    }
    return res;
  }
  factory Tensor.newI32(Context ctx, int value) => Tensor.fromPtr(gg.ggml_new_i32(ctx.ptr, value));
  factory Tensor.newF32(Context ctx, double value) => Tensor.fromPtr(gg.ggml_new_f32(ctx.ptr, value));
  factory Tensor.zeros(Context ctx, List<int> shape, {int type = GGML_TYPE_F32}) {
    final res = Tensor(ctx, type, shape);
    res.setZero();
    return res;
  }

  Tensor setZero() => Tensor.fromPtr(gg.ggml_set_zero(ptr));
  Tensor setI32(int value) => Tensor.fromPtr(gg.ggml_set_i32(ptr, value));
  Tensor setF32(double value) => Tensor.fromPtr(gg.ggml_set_f32(ptr, value));

  void setData(List<num> data) {
    switch (type) {
      case GGML_TYPE_I8:
        final p = this.data.cast<ffi.Int8>();
        for (var i = 0; i < data.length; i++) {
          p[i] = data[i].toInt();
        }
      case GGML_TYPE_I16:
        final p = this.data.cast<ffi.Int16>();
        for (var i = 0; i < data.length; i++) {
          p[i] = data[i].toInt();
        }
      case GGML_TYPE_I32:
        final p = this.data.cast<ffi.Int32>();
        for (var i = 0; i < data.length; i++) {
          p[i] = data[i].toInt();
        }
      case GGML_TYPE_I64:
        final p = this.data.cast<ffi.Int64>();
        for (var i = 0; i < data.length; i++) {
          p[i] = data[i].toInt();
        }
      case GGML_TYPE_F16:
        final p = this.data.cast<gg.ggml_fp16_t>();
        for (var i = 0; i < data.length; i++) {
          p[i] = data[i].toInt();
        }
      case GGML_TYPE_F32:
        final p = this.data.cast<ffi.Float>();
        for (var i = 0; i < data.length; i++) {
          p[i] = data[i].toDouble();
        }
      case GGML_TYPE_F64:
        final p = this.data.cast<ffi.Double>();
        for (var i = 0; i < data.length; i++) {
          p[i] = data[i].toDouble();
        }
      case _:
        throw UnimplementedError();
    }
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

  int get nelements => gg.ggml_nelements(ptr);
  int get nrows => gg.ggml_nrows(ptr);
  int get nbytes => gg.ggml_nbytes(ptr);
  int get nbytesPad => gg.ggml_nbytes_pad(ptr);

  /// unary or op name
  String get opDesc => gg.ggml_op_desc(ptr).cast<Utf8>().toDartString();

  int get elementSize => gg.ggml_element_size(ptr);
  bool get isTransposed => gg.ggml_is_transposed(ptr);
  bool get isContiguous => gg.ggml_is_contiguous(ptr);
  bool get isPermuted => gg.ggml_is_permuted(ptr);
  bool get isEmpty => gg.ggml_is_empty(ptr);
  bool get isScalar => gg.ggml_is_scalar(ptr);
  bool get isVector => gg.ggml_is_vector(ptr);
  bool get isMatrix => gg.ggml_is_matrix(ptr);
  bool get is3D => gg.ggml_is_3d(ptr);
  int get nDims => gg.ggml_n_dims(ptr);
  int get type => ref.type;

  bool areSameShape(Tensor other) => gg.ggml_are_same_shape(ptr, other.ptr);

  List<T> toList<T extends num>() {
    return List.generate(
      nelements,
      (i) => switch (type) {
        GGML_TYPE_I8 => data.cast<ffi.Int8>()[i] as T,
        GGML_TYPE_I16 => data.cast<ffi.Int16>()[i] as T,
        GGML_TYPE_I32 => data.cast<ffi.Int32>()[i] as T,
        GGML_TYPE_I64 => data.cast<ffi.Int64>()[i] as T,
        GGML_TYPE_F16 => data.cast<gg.ggml_fp16_t>()[i] as T,
        GGML_TYPE_F32 => data.cast<ffi.Float>()[i] as T,
        GGML_TYPE_F64 => data.cast<ffi.Double>()[i] as T,
        // GGML_TYPE_Q4_0 => data.cast<ffi.>()
        _ => throw UnimplementedError(),
      },
    );
  }

  // List<List<List<List<num>>>> toListND(){} TODO

  List<T> get<T extends num>() {
    switch (type) {
      case GGML_TYPE_I8:
        final cdata = calloc<ffi.Int8>(nelements);
        gg.ggml_backend_tensor_get(ptr, cdata.cast(), 0, nbytes);
        final res = List.generate(nelements, (i) => cdata[i] as T);
        calloc.free(cdata);
        return res;
      case GGML_TYPE_I16:
        final cdata = calloc<ffi.Int16>(nelements);
        gg.ggml_backend_tensor_get(ptr, cdata.cast(), 0, nbytes);
        final res = List.generate(nelements, (i) => cdata[i] as T);
        calloc.free(cdata);
        return res;
      case GGML_TYPE_I32:
        final cdata = calloc<ffi.Int32>(nelements);
        gg.ggml_backend_tensor_get(ptr, cdata.cast(), 0, nbytes);
        final res = List.generate(nelements, (i) => cdata[i] as T);
        calloc.free(cdata);
        return res;
      case GGML_TYPE_I64:
        final cdata = calloc<ffi.Int64>(nelements);
        gg.ggml_backend_tensor_get(ptr, cdata.cast(), 0, nbytes);
        final res = List.generate(nelements, (i) => cdata[i] as T);
        calloc.free(cdata);
        return res;
      case GGML_TYPE_F16:
        final cdata = calloc<gg.ggml_fp16_t>(nelements);
        gg.ggml_backend_tensor_get(ptr, cdata.cast(), 0, nbytes);
        final res = List.generate(nelements, (i) => cdata[i] as T);
        calloc.free(cdata);
        return res;
      case GGML_TYPE_F32:
        final cdata = calloc<ffi.Float>(nelements);
        gg.ggml_backend_tensor_get(ptr, cdata.cast(), 0, nbytes);
        final res = List.generate(nelements, (i) => cdata[i] as T);
        calloc.free(cdata);
        return res;
      case GGML_TYPE_F64:
        final cdata = calloc<ffi.Double>(nelements);
        gg.ggml_backend_tensor_get(ptr, cdata.cast(), 0, nbytes);
        final res = List.generate(nelements, (i) => cdata[i] as T);
        calloc.free(cdata);
        return res;
      default:
        throw UnsupportedError("Unsupported type");
    }
  }

  void set<T extends num>(List<num> data) {
    int size = 0;
    switch (type) {
      case GGML_TYPE_I8:
        final cdata = calloc<ffi.Int8>(data.length);
        for (var i = 0; i < data.length; i++) {
          cdata[i] = data[i].toInt();
        }
        size = ffi.sizeOf<ffi.Int8>() * data.length;
        gg.ggml_backend_tensor_set(ptr, cdata.cast(), 0, size);
      case GGML_TYPE_I16:
        final cdata = calloc<ffi.Int16>(data.length);
        for (var i = 0; i < data.length; i++) {
          cdata[i] = data[i].toInt();
        }
        size = ffi.sizeOf<ffi.Int16>() * data.length;
        gg.ggml_backend_tensor_set(ptr, cdata.cast(), 0, size);
      case GGML_TYPE_I32:
        final cdata = calloc<ffi.Int32>(data.length);
        for (var i = 0; i < data.length; i++) {
          cdata[i] = data[i].toInt();
        }
        size = ffi.sizeOf<ffi.Int32>() * data.length;
        gg.ggml_backend_tensor_set(ptr, cdata.cast(), 0, size);
      case GGML_TYPE_I64:
        final cdata = calloc<ffi.Int64>(data.length);
        for (var i = 0; i < data.length; i++) {
          cdata[i] = data[i].toInt();
        }
        size = ffi.sizeOf<ffi.Int64>() * data.length;
        gg.ggml_backend_tensor_set(ptr, cdata.cast(), 0, size);
      case GGML_TYPE_F16:
        final cdata = calloc<gg.ggml_fp16_t>(data.length);
        for (var i = 0; i < data.length; i++) {
          cdata[i] = data[i].toInt();
        }
        size = ffi.sizeOf<gg.ggml_fp16_t>() * data.length;
        gg.ggml_backend_tensor_set(ptr, cdata.cast(), 0, size);
      case GGML_TYPE_F32:
        final cdata = calloc<ffi.Float>(data.length);
        for (var i = 0; i < data.length; i++) {
          cdata[i] = data[i].toDouble();
        }
        size = ffi.sizeOf<ffi.Float>() * data.length;
        gg.ggml_backend_tensor_set(ptr, cdata.cast(), 0, size);
      case GGML_TYPE_F64:
        final cdata = calloc<ffi.Double>(data.length);
        for (var i = 0; i < data.length; i++) {
          cdata[i] = data[i].toDouble();
        }
        size = ffi.sizeOf<ffi.Double>() * data.length;
        gg.ggml_backend_tensor_set(ptr, cdata.cast(), 0, size);
      default:
        throw UnsupportedError("Unsupported type");
    }
  }
  // struct ggml_tensor * ggml_format_name(      struct ggml_tensor * tensor, const char * fmt, ...);

  // static final finalizer = ffi.NativeFinalizer(gg.ggmltensor);

  @override
  gg.ggml_tensor get ref => ptr.ref;

  @override
  String toString() {
    return "Tensor(address=0x${ptr.address.toRadixString(16)})";
  }
}
