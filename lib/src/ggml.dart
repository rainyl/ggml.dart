import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'ggml.g.dart' as gg;
import 'tensor.dart';

int ggml_graph_overhead() => gg.ggml_graph_overhead();

int ggml_tensor_overhead() => gg.ggml_tensor_overhead();

int ggml_blck_size(int type) => gg.ggml_blck_size(type);

/// size in bytes for all elements in a block
int ggml_type_size(int type) => gg.ggml_type_size(type);

/// size in bytes for all elements in a row
int ggmlRowSize(int type, int ne) => gg.ggml_row_size(type, ne);

String ggmlTypeName(int type) => gg.ggml_type_name(type).cast<Utf8>().toDartString();
String ggmlOpName(int op) => gg.ggml_op_name(op).cast<Utf8>().toDartString();
String ggmlOpSymbol(int op) => gg.ggml_op_symbol(op).cast<Utf8>().toDartString();

String ggmlUnaryOpName(int op) => gg.ggml_unary_op_name(op).cast<Utf8>().toDartString();

bool ggmlIsQuantized(int type) => gg.ggml_is_quantized(type);

int ggmlFtypeToGgmlType(int type) => gg.ggml_ftype_to_ggml_type(type);

int ggmlFp32ToFp16(double fp) => gg.ggml_fp32_to_fp16(fp);
double ggmlFp16ToFp32(int fp) => gg.ggml_fp16_to_fp32(fp);

List<int> ggmlFp32ToFp16Row(List<double> x) {
  final cx = calloc<ffi.Float>(x.length);
  for (var i = 0; i < x.length; i++) {
    cx[i] = x[i];
  }
  final cy = calloc<gg.ggml_fp16_t>(x.length);
  gg.ggml_fp32_to_fp16_row(cx, cy, x.length);
  calloc.free(cx);
  final y = List.generate(x.length, (i) => cy[i]);
  calloc.free(cy);
  return y;
}

List<num> ggml_tensor_get(Tensor tensor) => tensor.tensorGet();
void ggml_tensor_set(Tensor tensor, List<num> data) => tensor.tensorSet(data);
