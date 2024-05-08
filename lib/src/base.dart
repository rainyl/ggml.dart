// ignore_for_file: camel_case_types

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:equatable/equatable.dart';

import 'ggml.g.dart' as gg;

abstract class GGBase<T extends ffi.NativeType> with EquatableMixin implements ffi.Finalizable {
  GGBase.fromPtr(this.ptr);
  ffi.Pointer<T> ptr;

  bool get isNull => ptr == ffi.nullptr;

  @override
  List<Object> get props => [ptr.address];
}

abstract class GGStruct<T extends ffi.Struct> extends GGBase<T> {
  GGStruct.fromPtr(super.ptr) : super.fromPtr();

  T get ref => throw UnimplementedError();
}

typedef NativeFinalizerFunctionT<T extends ffi.NativeType>
    = ffi.Pointer<ffi.NativeFunction<ffi.Void Function(T token)>>;

ffi.NativeFinalizer ggFinalizer<T extends ffi.NativeType>(NativeFinalizerFunctionT<T> func) =>
    ffi.NativeFinalizer(func.cast<ffi.NativeFinalizerFunction>());

class GGObject extends GGStruct<gg.ggml_object> {
  GGObject._(super.ptr) : super.fromPtr() {
    // finalizer.attach(this, ptr.cast());
  }

  factory GGObject(int offs, int size, GGObject next, int type, List<int> padding) {
    assert(padding.length == 4);
    final pPadding = ffi.Array<ffi.Char>(padding.length);
    for (int i = 0; i < padding.length; i++) {
      pPadding[i] = padding[i];
    }
    final p = calloc<gg.ggml_object>()
      ..ref.offs = offs
      ..ref.size = size
      ..ref.type = type
      ..ref.next = next.ptr
      ..ref.padding = pPadding;
    return GGObject._(p);
  }

  int get offs => ref.offs;
  int get size => ref.size;
  int get type => ref.type;

  // static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  @override
  gg.ggml_object get ref => ptr.ref;
}

class Scratch extends GGStruct<gg.ggml_scratch> {
  Scratch.fromPtr(super.ptr) : super.fromPtr() {
    // finalizer.attach(this, ptr.cast());
  }

  factory Scratch({int? offs, int? size, ffi.Pointer<ffi.Void>? data}) {
    final p = calloc<gg.ggml_scratch>()
      ..ref.offs = offs ?? 0
      ..ref.size = size ?? 0
      ..ref.data = data ?? ffi.nullptr;
    return Scratch.fromPtr(p);
  }

  int get offs => ref.offs;
  int get size => ref.size;
  ffi.Pointer<ffi.Void> get data => ref.data;

  // static final finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  @override
  gg.ggml_scratch get ref => ptr.ref;
}
