import 'dart:ffi' as ffi;

import 'package:equatable/equatable.dart';

abstract class GgObject<T extends ffi.NativeType> implements ffi.Finalizable {
  GgObject.fromPtr(this.ptr);
  ffi.Pointer<T> ptr;
}

abstract class GgStruct<T extends ffi.Struct> extends GgObject<T> with EquatableMixin {
  GgStruct.fromPtr(super.ptr) : super.fromPtr();

  T get ref => throw UnimplementedError();

  @override
  List<int> get props => [ptr.address];
}

typedef NativeFinalizerFunctionT<T extends ffi.NativeType>
    = ffi.Pointer<ffi.NativeFunction<ffi.Void Function(T token)>>;

ffi.NativeFinalizer ggFinalizer<T extends ffi.NativeType>(NativeFinalizerFunctionT<T> func) =>
    ffi.NativeFinalizer(func.cast<ffi.NativeFinalizerFunction>());
