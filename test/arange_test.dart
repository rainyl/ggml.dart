import 'package:test/test.dart';
import 'package:ggml/ggml.dart' as gg;

void main() {
  test('arange', () {
    const int numTensors = 2;
    final backend = gg.Backend(gg.GGML_BACKEND_TYPE_CPU);
    final overhead = gg.ggml_tensor_overhead();
    final params = gg.InitParams(memSize: overhead * numTensors + 2 * 1024 * 1024, noAlloc: true);
    final ctx = gg.Context.init(params);
    final t = ctx.arange(0, 3, 1);
    expect(t.ne.first, 3);

    final galloc = gg.GraphAllocr(backend.getDefaultBufferType());
    final graph = gg.CGraph(ctx);
    ctx.buildForwardExpand(graph, t);
    galloc.allocGraph(graph);

    int nThreads = 4;
    backend.cpuSetNThreads(nThreads);
    backend.graphCompute(graph);

    print(t.ne);
    print(List.generate(t.ne.first, (i)=>t.getF32_1D(i)));
  });
}
