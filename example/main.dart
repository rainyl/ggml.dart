import 'package:ggml/ggml.dart' as gg;

void main() {
  const int numTensors = 2;
    final backend = gg.GGMLBackend(gg.GGML_BACKEND_TYPE_CPU);
    final overhead = gg.ggmlTensorOverhead();
    final params = gg.GGMLInitParams(memSize: overhead * numTensors + 2 * 1024 * 1024, noAlloc: true);
    final ctx = gg.GGMLContext.init(params);
    
    final t = ctx.arange(0, 3, 1);
    print("t.ne: ${t.ne}");

    final galloc = gg.GGMLGAllocr(backend.getDefaultBufferType());
    final graph = gg.GGMLCGraph(ctx);
    ctx.buildForwardExpand(graph, t);
    galloc.allocGraph(graph);

    int nThreads = 4;
    backend.setNThreads(nThreads);
    backend.graphCompute(graph);

    print(List.generate(t.ne.first, (i)=>t.getF32_1D(i)));
}