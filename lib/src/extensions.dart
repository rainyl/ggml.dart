import 'dart:ffi' as ffi;

extension PointerExtensions<T extends ffi.NativeType> on ffi.Pointer<T> {
  bool operator >(ffi.Pointer<T> other) => address > other.address;

  bool operator <(ffi.Pointer<T> other) => address < other.address;

  bool operator >=(ffi.Pointer<T> other) => address >= other.address;

  bool operator <=(ffi.Pointer<T> other) => address <= other.address;
}
