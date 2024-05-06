import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'ggml.g.dart' as gg;

class InitParams extends GGStruct<gg.ggml_init_params> {
  InitParams._(super.ptr) : super.fromPtr() {
    finalizer.attach(this, ptr.cast());
  }

  factory InitParams({int? memSize, bool? noAlloc, ffi.Pointer<ffi.Void>? memBuffer}) {
    final p = calloc<gg.ggml_init_params>()
      ..ref.mem_size = memSize ?? 0
      ..ref.mem_buffer = memBuffer ?? ffi.nullptr
      ..ref.no_alloc = noAlloc ?? false;
    return InitParams._(p);
  }

  int get memSize => ref.mem_size;
  bool get noAlloc => ref.no_alloc;
  ffi.Pointer<ffi.Void> get memBuffer => ref.mem_buffer;

  static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  @override
  gg.ggml_init_params get ref => ptr.ref;
}

class OptParams extends GGStruct<gg.ggml_opt_params> {
  OptParams._(super.ptr) : super.fromPtr() {
    // finalizer.attach(this, ptr.cast());
  }

  factory OptParams(int type) {
    final s = gg.ggml_opt_default_params(type);
    final p = calloc<gg.ggml_opt_params>()..ref = s;
    return OptParams._(p);
  }

  // static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);
  @override
  gg.ggml_opt_params get ref => ptr.ref;
}
