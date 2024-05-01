import 'ggml.g.dart' as gg;

// - ggml_quantize_init can be called multiple times with the same type
//   it will only initialize the quantization tables for the first call or after ggml_quantize_free
//   automatically called by ggml_quantize_chunk for convenience
//
// - ggml_quantize_free will free any memory allocated by ggml_quantize_init
//   call this at the end of the program to avoid memory leaks
//
// note: these are thread-safe
//

void ggmlQuantizeInit(int type) => gg.ggml_quantize_init(type);

void ggmlQuantizeFree() => gg.ggml_quantize_free();

// some quantization type cannot be used without an importance matrix
bool ggmlQuantizeRequiresImatrix(int type) => gg.ggml_quantize_requires_imatrix(type);

// TODO:
// calls ggml_quantize_init internally (i.e. can allocate memory)
// int ggmlQuantizeChunk(
//   int type,
//   List<double> src,
//   dst,
//   int start,
//   int nrows,
//   int nPerRow,
//   List<double> imatrix,
// ) {}
