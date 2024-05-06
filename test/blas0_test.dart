import 'dart:math';
import 'package:test/test.dart';
import 'package:ggml/ggml.dart' as gg;

const int nThreads = 1;

List<double> mulMatF32(List<double> src0, List<double> src1, int m, int n, int k) {
  final dst = List.filled(m * n, 0.0);
  for (var i = 0; i < m; i++) {
    for (var j = 0; j < n; j++) {
      var sum = 0.0;
      for (var l = 0; l < k; l++) {
        sum += src0[i * k + l] * src1[j * k + l];
      }
      dst[j * m + i] = sum;
    }
  }
  return dst;
}

void main() {
  final random = Random();
  final M = Random().nextInt(3000) + 1;
  final N = Random().nextInt(3000) + 1;
  final K = Random().nextInt(3000) + 1;
  print("M=$M, N=$N, K=$K");

  final params = gg.InitParams(memSize: 2048 * 1024 * 1024, noAlloc: false);
  final ctx0 = gg.Context.init(params);

  final src0 = List.generate(M * K, (i) => random.nextDouble() * 1e-3);
  final src1 = List.generate(N * K, (i) => random.nextDouble() * 1e-3);

  final s0F32 = gg.Tensor.new2D(ctx0, gg.GGML_TYPE_F32, K, M, data: src0);
  final s1F32 = gg.Tensor.new2D(ctx0, gg.GGML_TYPE_F32, K, N, data: src1);

  final s0F16 = gg.Tensor.new2D(ctx0, gg.GGML_TYPE_F16, K, M, data: src0);
  final s1F16 = gg.Tensor.new2D(ctx0, gg.GGML_TYPE_F16, K, N, data: src1);

  final dst0 = mulMatF32(src0, src1, M, N, K);

  test('ggml', () {
    final dst2 = ctx0.mulMat(s0F32, s1F32);
    final gf = gg.CGraph(ctx0);
    gf.buildForwardExpand(dst2);
    expect(gf.computeWithCtx(ctx0, nThreads), gg.GGML_STATUS_SUCCESS);
    expect(dst2.toList<double>().indexed.map((e) => dst0[e.$1] - e.$2 < 1e-6).every((e) => e), true);

    final dst3 = ctx0.mulMat(s0F16, s1F16);
    final gf1 = gg.CGraph(ctx0);
    gf1.buildForwardExpand(dst3);
    expect(gf1.computeWithCtx(ctx0, nThreads), gg.GGML_STATUS_SUCCESS);
    expect(dst3.toList<double>().indexed.map((e) => dst0[e.$1] - e.$2 < 1e-3).every((e) => e), true);
  });
}
