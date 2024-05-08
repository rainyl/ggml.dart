// ignore_for_file: non_constant_identifier_names, constant_identifier_names

import 'package:test/test.dart';
import 'package:ggml/ggml.dart' as gg;

class TestModel {
  TestModel(this.a, this.b, this.backend, this.buffer, this.ctx);

  factory TestModel.load([bool useGpu = false]) {
    const K = 3, IC = 10, OC = 10;
    const IL = 8, N = 1;
    final adata = List.generate(K * IC * OC, (i) => 4.5);
    final hadata = gg.ggmlFp32ToFp16Row(adata);
    final bdata = List.generate(IL * IC * N, (i) => 2.5);

    var bufferSize = K * IC * OC * gg.ggml_type_size(gg.GGML_TYPE_F16);
    bufferSize += IL * IC * N * gg.ggml_type_size(gg.GGML_TYPE_F32);
    bufferSize += 1024;

    int numTensors = 2;
    final params = gg.InitParams(memSize: gg.ggml_tensor_overhead() * numTensors, noAlloc: true);
    final backend = gg.Backend.cpuInit();
    final buffer = backend.allocBuffer(bufferSize);

    final ctx = gg.Context.init(params);
    final a = gg.Tensor.new3D(ctx, gg.GGML_TYPE_F16, K, IC, OC);
    final b = gg.Tensor.new3D(ctx, gg.GGML_TYPE_F32, IL, IC, N);

    final alloc = gg.TensorAllocr(buffer);
    alloc.alloc(a);
    if (backend.isCpu()) {
      a.setData(hadata);
    } else {
      backend.tensorSet(a, hadata);
    }

    alloc.alloc(b);
    if (backend.isCpu()) {
      b.setData(bdata);
    } else {
      backend.tensorSet(b, bdata);
    }

    return TestModel(a, b, backend, buffer, ctx);
  }

  gg.Tensor a;
  gg.Tensor b;
  gg.Backend backend;
  gg.BackendBuffer buffer;
  gg.Context ctx;

  int nThreads = 1;

  gg.CGraph buildGraph() {
    final bufSize = gg.ggml_tensor_overhead() * gg.GGML_DEFAULT_GRAPH_SIZE + gg.ggml_graph_overhead();

    final params0 = gg.InitParams(memSize: bufSize, noAlloc: true);
    final ctx0 = gg.Context.init(params0);
    final gf = ctx0.newGraph();

    const int s0 = 1, p0 = 1, d0 = 1;

    final im2col_0 = ctx0.im2col(a, b, s0, 0, p0, 0, d0, 0, false, gg.GGML_TYPE_F16);
    im2col_0.name = 'im2col_res';
    gf.buildForwardExpand(im2col_0);

    final conv1d_res = ctx0.conv1d(a, b, s0, p0, d0);
    conv1d_res.name = 'conv1d_res';
    gf.buildForwardExpand(conv1d_res);
    return gf;
  }

  gg.CGraph computeGraph(gg.GraphAllocr allocr) {
    final gf = buildGraph();
    allocr.allocGraph(gf);

    if (backend.isCpu()) {
      backend.cpuSetNThreads(nThreads);
    }

    backend.graphCompute(gf);

    return gf;
  }
}

void main() {
  test('conv1d', () {
    final model = TestModel.load();
    final allocr = gg.GraphAllocr(model.backend.getDefaultBufferType());

    final gf = model.buildGraph();
    allocr.reserve(gf);
    final memSize = allocr.getBufferSize(0);
    print("compute buffer size: ${memSize / 1024.0 / 1024.0} MB");

    final gf_res = model.computeGraph(allocr);
    gg.Tensor? im2col_res, conv1d_res;
    for (var i = 0; i < gf_res.nNodes; i++) {
      print("node $i: ${gf_res.getNode(i).name}");
      if (gf_res.getNode(i).name == "im2col_res") {
        im2col_res = gf_res.getNode(i);
      } else if (gf_res.getNode(i).name == "conv1d_res") {
        conv1d_res = gf_res.getNode(i);
      }
    }
    expect(im2col_res, isNotNull);
    expect(conv1d_res, isNotNull);

    final im2col_data = im2col_res!.tensorGet();
    final conv2d_data = conv1d_res!.tensorGet();

    final expected_conv1d = [
      225.00,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      225.00,
      225.00,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      225.00,
      225.00,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      225.00,
      225.00,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      225.00,
      225.00,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      225.00,
      225.00,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      225.00,
      225.00,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      225.00,
      225.00,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      225.00,
      225.00,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      225.00,
      225.00,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      337.50,
      225.00
    ];

    final expected_im2col = [
      0,
      16640,
      16640,
      0,
      16640,
      16640,
      0,
      16640,
      16640,
      0,
      16640,
      16640,
      0,
      16640,
      16640,
      0,
      16640,
      16640,
      0,
      16640,
      16640,
      0,
      16640,
      16640,
      0,
      16640,
      16640,
      0,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640,
      16640
    ];

    expect(expected_im2col.indexed.map((e) => e.$2 == im2col_data[e.$1]).every((e) => e), true);
    expect(expected_conv1d.indexed.map((e) => e.$2 == conv2d_data[e.$1]).every((e) => e), true);
  });
}
