import 'package:test/test.dart';
import 'package:ggml/ggml.dart' as gg;

const int nThreads = 1;

gg.Context makeCtx() {
  final params = gg.InitParams(memSize: 2048 * 1024 * 1024);
  return gg.Context.init(params);
}

void main() {
  test('test_conv_transpose_1d', () {
    final bufF32 = List.generate(1024, (i) => i.toDouble());
    final bufF16 = bufF32.map((e) => gg.ggmlFp32ToFp16(e)).toList();

    final expectedOut1 = [
      [18.0, 45.0, 59.0, 37.0],
      [24.0, 61.0, 83.0, 51.0],
      [30.0, 77.0, 107.0, 65.0],
    ];
    final expectedOut2 = [
      [18.0, 21.0, 24.0, 29.0, 30.0, 37.0],
      [24.0, 27.0, 34.0, 39.0, 44.0, 51.0],
      [30.0, 33.0, 44.0, 49.0, 58.0, 65.0],
    ];
    final expectedOut3 = [
      [18.0, 21.0, 0.0, 24.0, 29.0, 0.0, 30.0, 37.0],
      [24.0, 27.0, 0.0, 34.0, 39.0, 0.0, 44.0, 51.0],
      [30.0, 33.0, 0.0, 44.0, 49.0, 0.0, 58.0, 65.0],
    ];

    final ctx = makeCtx();
    final t = gg.Tensor.new2D(ctx, gg.GGML_TYPE_F32, 3, 2, data: bufF32); // l x cin

    final k = gg.Tensor.new3D(ctx, gg.GGML_TYPE_F16, 2, 3, 2, data: bufF16); // k x cout x cin

    final out1 = ctx.convTranspose1d(k, t, 1, 0, 1);
    final out2 = ctx.convTranspose1d(k, t, 2, 0, 1);
    final out3 = ctx.convTranspose1d(k, t, 3, 0, 1);

    final gf1 = ctx.newGraph();
    final gf2 = ctx.newGraph();
    final gf3 = ctx.newGraph();

    ctx.buildForwardExpand(gf1, out1);
    ctx.buildForwardExpand(gf2, out2);
    ctx.buildForwardExpand(gf3, out3);

    ctx.graphComputeWithCtx(gf1, nThreads);
    ctx.graphComputeWithCtx(gf2, nThreads);
    ctx.graphComputeWithCtx(gf3, nThreads);

    expect(out1.ne, [4, 3, 1, 1]);
    expect(out1.dtype, gg.GGML_TYPE_F32);
    expect(out1.toList2D(), expectedOut1);

    expect(out2.ne, [6, 3, 1, 1]);
    expect(out2.dtype, gg.GGML_TYPE_F32);
    expect(out2.toList2D(), expectedOut2);

    expect(out3.ne, [8, 3, 1, 1]);
    expect(out3.dtype, gg.GGML_TYPE_F32);
    expect(out3.toList2D(), expectedOut3);
  });

  test('test_conv_transpose_2d', () {
    final bufF32 = List.generate(1024, (i) => i.toDouble());
    final bufF16 = bufF32.map((e) => gg.ggmlFp32ToFp16(e)).toList();

    final expectedOut1 = [
      [
        [72.0, 162.0, 188.0, 106.0],
        [192.0, 430.0, 490.0, 274.0],
        [132.0, 292.0, 326.0, 180.0],
      ],
      [
        [96.0, 218.0, 260.0, 146.0],
        [264.0, 590.0, 682.0, 378.0],
        [180.0, 396.0, 446.0, 244.0],
      ],
      [
        [120.0, 274.0, 332.0, 186.0],
        [336.0, 750.0, 874.0, 482.0],
        [228.0, 500.0, 566.0, 308.0],
      ],
    ];

    final expectedOut2 = [
      [
        [72.0, 78.0, 84.0, 92.0, 96.0, 106.0],
        [84.0, 90.0, 100.0, 108.0, 116.0, 126.0],
        [108.0, 120.0, 120.0, 134.0, 132.0, 148.0],
        [132.0, 144.0, 148.0, 162.0, 164.0, 180.0],
      ],
      [
        [96.0, 102.0, 116.0, 124.0, 136.0, 146.0],
        [108.0, 114.0, 132.0, 140.0, 156.0, 166.0],
        [156.0, 168.0, 176.0, 190.0, 196.0, 212.0],
        [180.0, 192.0, 204.0, 218.0, 228.0, 244.0],
      ],
      [
        [120.0, 126.0, 148.0, 156.0, 176.0, 186.0],
        [132.0, 138.0, 164.0, 172.0, 196.0, 206.0],
        [204.0, 216.0, 232.0, 246.0, 260.0, 276.0],
        [228.0, 240.0, 260.0, 274.0, 292.0, 308.0],
      ],
    ];

    final expectedOut3 = [
      [
        [72.0, 78.0, 0.0, 84.0, 92.0, 0.0, 96.0, 106.0],
        [84.0, 90.0, 0.0, 100.0, 108.0, 0.0, 116.0, 126.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [108.0, 120.0, 0.0, 120.0, 134.0, 0.0, 132.0, 148.0],
        [132.0, 144.0, 0.0, 148.0, 162.0, 0.0, 164.0, 180.0],
      ],
      [
        [96.0, 102.0, 0.0, 116.0, 124.0, 0.0, 136.0, 146.0],
        [108.0, 114.0, 0.0, 132.0, 140.0, 0.0, 156.0, 166.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [156.0, 168.0, 0.0, 176.0, 190.0, 0.0, 196.0, 212.0],
        [180.0, 192.0, 0.0, 204.0, 218.0, 0.0, 228.0, 244.0],
      ],
      [
        [120.0, 126.0, 0.0, 148.0, 156.0, 0.0, 176.0, 186.0],
        [132.0, 138.0, 0.0, 164.0, 172.0, 0.0, 196.0, 206.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [204.0, 216.0, 0.0, 232.0, 246.0, 0.0, 260.0, 276.0],
        [228.0, 240.0, 0.0, 260.0, 274.0, 0.0, 292.0, 308.0],
      ],
    ];

    final ctx = makeCtx();
    final t = gg.Tensor.new4D(ctx, gg.GGML_TYPE_F32, 3, 2, 2, 1, data: bufF32); // w x h x cin
    final k = gg.Tensor.new4D(ctx, gg.GGML_TYPE_F16, 2, 2, 3, 2, data: bufF16); // w x h cin x cout

    final out1 = ctx.convTranspose2dP0(k, t, 1);
    final out2 = ctx.convTranspose2dP0(k, t, 2);
    final out3 = ctx.convTranspose2dP0(k, t, 3);

    final gf1 = ctx.newGraph();
    final gf2 = ctx.newGraph();
    final gf3 = ctx.newGraph();

    ctx.buildForwardExpand(gf1, out1);
    ctx.buildForwardExpand(gf2, out2);
    ctx.buildForwardExpand(gf3, out3);

    ctx.graphComputeWithCtx(gf1, nThreads);
    ctx.graphComputeWithCtx(gf2, nThreads);
    ctx.graphComputeWithCtx(gf3, nThreads);

    expect(out1.ne, [4, 3, 3, 1]);
    expect(out1.dtype, gg.GGML_TYPE_F32);
    expect(out1.toList3D(), expectedOut1);

    expect(out2.ne, [6, 4, 3, 1]);
    expect(out2.dtype, gg.GGML_TYPE_F32);
    expect(out2.toList3D(), expectedOut2);

    expect(out3.ne, [8, 5, 3, 1]);
    expect(out3.dtype, gg.GGML_TYPE_F32);
    expect(out3.toList3D(), expectedOut3);
  });
}
