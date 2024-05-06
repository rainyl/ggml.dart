import 'package:ggml/ggml.dart' as gg;
import 'package:test/test.dart';

void main() {
  test('System Info', () {
    expect(gg.cpuHasArmFma(), isA<int>());
    expect(gg.cpuHasAvx(), isA<int>());
    expect(gg.cpuHasAvxVnni(), isA<int>());
    expect(gg.cpuHasAvx2(), isA<int>());
    expect(gg.cpuHasAvx512(), isA<int>());
    expect(gg.cpuHasAvx512Vbmi(), isA<int>());
    expect(gg.cpuHasAvx512Vnni(), isA<int>());
    expect(gg.cpuHasFma(), isA<int>());
    expect(gg.cpuHasNeon(), isA<int>());
    expect(gg.cpuHasMetal(), isA<int>());
    expect(gg.cpuHasF16c(), isA<int>());
    expect(gg.cpuHasFp16Va(), isA<int>());
    expect(gg.cpuHasWasmSimd(), isA<int>());
    expect(gg.cpuHasBlas(), isA<int>());
    expect(gg.cpuHasCuda(), isA<int>());
    expect(gg.cpuHasClblast(), isA<int>());
    expect(gg.cpuHasVulkan(), isA<int>());
    expect(gg.cpuHasKompute(), isA<int>());
    expect(gg.cpuHasGpublas(), isA<int>());
    expect(gg.cpuHasSse3(), isA<int>());
    expect(gg.cpuHasSsse3(), isA<int>());
    expect(gg.cpuHasSycl(), isA<int>());
    expect(gg.cpuHasVsx(), isA<int>());
    expect(gg.cpuHasMatmulInt8(), isA<int>());
  });
}
