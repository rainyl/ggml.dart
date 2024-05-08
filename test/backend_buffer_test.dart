import 'dart:ffi' as ffi;
import 'package:ggml/ggml.dart' as gg;
import 'package:test/test.dart';

bool isPow2(int x) {
  return (x & (x - 1)) == 0;
}

void testBuffer(gg.Backend backend, gg.BackendBufferType buft) {
  expect(backend.getDefaultBufferType(), buft);
  expect(buft.supportsBackend(backend), true);

  final buffer = buft.allocBuffer(1024);
  expect(buffer.ptr, isNotNull);
  expect(isPow2(buffer.getAlignment()), true);
  expect(buffer.getBase(), isNotNull);
  expect(buffer.getSize(), greaterThanOrEqualTo(1024));

  final params = gg.InitParams(memSize: 1024, noAlloc: true);
  final ctx = gg.Context.init(params);
  final n = 10;
  final tensor = gg.Tensor.new1D(ctx, gg.GGML_TYPE_F32, n);

  expect(buffer.getAllocSize(tensor), greaterThanOrEqualTo(n * ffi.sizeOf<ffi.Float>()));

  final allocr = gg.TensorAllocr(buffer);
  allocr.alloc(tensor);

  expect(tensor.data, isNotNull);
  expect(tensor.data, greaterThanOrEqualTo(buffer.getBase()));

  final data = List.generate(n, (i) => i.toDouble());
  backend.tensorSet(tensor, data);
  final data2 = backend.tensorGet(tensor);
  expect(data2, data);
}

void main() {
  for (int i = 0; i < gg.ggmlBackendRegGetCount(); i++) {
    final backend = gg.Backend.regInitBackend(i);
    print(backend.name());

    test(backend.name(), () => testBuffer(backend, gg.Backend.regGetDefaultBufferType(i)));
  }
}
