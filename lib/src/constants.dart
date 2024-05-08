// ignore_for_file: constant_identifier_names

const int GGML_BACKEND_BUFFER_USAGE_ANY = 0;
const int GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1;

const int GGML_BACKEND_TYPE_CPU = 0;
const int GGML_BACKEND_TYPE_GPU = 10;
const int GGML_BACKEND_TYPE_GPU_SPLIT = 20;

const int GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0;
const int GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT = 1;
const int GGML_CGRAPH_EVAL_ORDER_COUNT = 2;

/// model file types

const int GGML_FTYPE_UNKNOWN = -1;
const int GGML_FTYPE_ALL_F32 = 0;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_F16 = 1;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_Q4_0 = 2;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_Q4_1 = 3;

/// tok_embeddings.weight and output.weight are F16
const int GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_Q8_0 = 7;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_Q5_0 = 8;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_Q5_1 = 9;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_Q2_K = 10;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_Q3_K = 11;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_Q4_K = 12;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_Q5_K = 13;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_Q6_K = 14;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_IQ2_XXS = 15;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_IQ2_XS = 16;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_IQ3_XXS = 17;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_IQ1_S = 18;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_IQ4_NL = 19;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_IQ3_S = 20;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_IQ2_S = 21;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_IQ4_XS = 22;

/// except 1d tensors
const int GGML_FTYPE_MOSTLY_IQ1_M = 23;

/// linesearch methods

const int GGML_LINESEARCH_DEFAULT = 1;
const int GGML_LINESEARCH_BACKTRACKING_ARMIJO = 0;
const int GGML_LINESEARCH_BACKTRACKING_WOLFE = 1;
const int GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2;

const int GGML_LOG_LEVEL_ERROR = 2;
const int GGML_LOG_LEVEL_WARN = 3;
const int GGML_LOG_LEVEL_INFO = 4;
const int GGML_LOG_LEVEL_DEBUG = 5;

/// numa strategies

const int GGML_NUMA_STRATEGY_DISABLED = 0;
const int GGML_NUMA_STRATEGY_DISTRIBUTE = 1;
const int GGML_NUMA_STRATEGY_ISOLATE = 2;
const int GGML_NUMA_STRATEGY_NUMACTL = 3;
const int GGML_NUMA_STRATEGY_MIRROR = 4;
const int GGML_NUMA_STRATEGY_COUNT = 5;

const int GGML_OBJECT_TYPE_TENSOR = 0;
const int GGML_OBJECT_TYPE_GRAPH = 1;
const int GGML_OBJECT_TYPE_WORK_BUFFER = 2;

/// available tensor operations:

const int GGML_OP_NONE = 0;
const int GGML_OP_DUP = 1;
const int GGML_OP_ADD = 2;
const int GGML_OP_ADD1 = 3;
const int GGML_OP_ACC = 4;
const int GGML_OP_SUB = 5;
const int GGML_OP_MUL = 6;
const int GGML_OP_DIV = 7;
const int GGML_OP_SQR = 8;
const int GGML_OP_SQRT = 9;
const int GGML_OP_LOG = 10;
const int GGML_OP_SUM = 11;
const int GGML_OP_SUM_ROWS = 12;
const int GGML_OP_MEAN = 13;
const int GGML_OP_ARGMAX = 14;
const int GGML_OP_REPEAT = 15;
const int GGML_OP_REPEAT_BACK = 16;
const int GGML_OP_CONCAT = 17;
const int GGML_OP_SILU_BACK = 18;

/// normalize
const int GGML_OP_NORM = 19;
const int GGML_OP_RMS_NORM = 20;
const int GGML_OP_RMS_NORM_BACK = 21;
const int GGML_OP_GROUP_NORM = 22;
const int GGML_OP_MUL_MAT = 23;
const int GGML_OP_MUL_MAT_ID = 24;
const int GGML_OP_OUT_PROD = 25;
const int GGML_OP_SCALE = 26;
const int GGML_OP_SET = 27;
const int GGML_OP_CPY = 28;
const int GGML_OP_CONT = 29;
const int GGML_OP_RESHAPE = 30;
const int GGML_OP_VIEW = 31;
const int GGML_OP_PERMUTE = 32;
const int GGML_OP_TRANSPOSE = 33;
const int GGML_OP_GET_ROWS = 34;
const int GGML_OP_GET_ROWS_BACK = 35;
const int GGML_OP_DIAG = 36;
const int GGML_OP_DIAG_MASK_INF = 37;
const int GGML_OP_DIAG_MASK_ZERO = 38;
const int GGML_OP_SOFT_MAX = 39;
const int GGML_OP_SOFT_MAX_BACK = 40;
const int GGML_OP_ROPE = 41;
const int GGML_OP_ROPE_BACK = 42;
const int GGML_OP_ALIBI = 43;
const int GGML_OP_CLAMP = 44;
const int GGML_OP_CONV_TRANSPOSE_1D = 45;
const int GGML_OP_IM2COL = 46;
const int GGML_OP_CONV_TRANSPOSE_2D = 47;
const int GGML_OP_POOL_1D = 48;
const int GGML_OP_POOL_2D = 49;

/// nearest interpolate
const int GGML_OP_UPSCALE = 50;
const int GGML_OP_PAD = 51;
const int GGML_OP_ARANGE = 52;
const int GGML_OP_TIMESTEP_EMBEDDING = 53;
const int GGML_OP_ARGSORT = 54;
const int GGML_OP_LEAKY_RELU = 55;
const int GGML_OP_FLASH_ATTN = 56;
const int GGML_OP_FLASH_FF = 57;
const int GGML_OP_FLASH_ATTN_BACK = 58;
const int GGML_OP_SSM_CONV = 59;
const int GGML_OP_SSM_SCAN = 60;
const int GGML_OP_WIN_PART = 61;
const int GGML_OP_WIN_UNPART = 62;
const int GGML_OP_GET_REL_POS = 63;
const int GGML_OP_ADD_REL_POS = 64;
const int GGML_OP_UNARY = 65;
const int GGML_OP_MAP_UNARY = 66;
const int GGML_OP_MAP_BINARY = 67;
const int GGML_OP_MAP_CUSTOM1_F32 = 68;
const int GGML_OP_MAP_CUSTOM2_F32 = 69;
const int GGML_OP_MAP_CUSTOM3_F32 = 70;
const int GGML_OP_MAP_CUSTOM1 = 71;
const int GGML_OP_MAP_CUSTOM2 = 72;
const int GGML_OP_MAP_CUSTOM3 = 73;
const int GGML_OP_CROSS_ENTROPY_LOSS = 74;
const int GGML_OP_CROSS_ENTROPY_LOSS_BACK = 75;
const int GGML_OP_COUNT = 76;

const int GGML_OP_POOL_MAX = 0;
const int GGML_OP_POOL_AVG = 1;
const int GGML_OP_POOL_COUNT = 2;

/// optimization return values

const int GGML_OPT_RESULT_OK = 0;
const int GGML_OPT_RESULT_DID_NOT_CONVERGE = 1;
const int GGML_OPT_RESULT_NO_CONTEXT = 2;
const int GGML_OPT_RESULT_INVALID_WOLFE = 3;
const int GGML_OPT_RESULT_FAIL = 4;
const int GGML_OPT_RESULT_CANCEL = 5;
const int GGML_LINESEARCH_FAIL = -128;
const int GGML_LINESEARCH_MINIMUM_STEP = -127;
const int GGML_LINESEARCH_MAXIMUM_STEP = -126;
const int GGML_LINESEARCH_MAXIMUM_ITERATIONS = -125;
const int GGML_LINESEARCH_INVALID_PARAMETERS = -124;

/// optimization methods

const int GGML_OPT_TYPE_ADAM = 0;
const int GGML_OPT_TYPE_LBFGS = 1;

/// precision

const int GGML_PREC_DEFAULT = 0;
const int GGML_PREC_F32 = 1;

/// sort rows

const int GGML_SORT_ORDER_ASC = 0;
const int GGML_SORT_ORDER_DESC = 1;

const int GGML_STATUS_ALLOC_FAILED = -2;
const int GGML_STATUS_FAILED = -1;
const int GGML_STATUS_SUCCESS = 0;
const int GGML_STATUS_ABORTED = 1;

/// NOTE: the INIT or FINALIZE pass is not scheduled unless explicitly enabled.
/// This behavior was changed since https://github.com/ggerganov/llama.cpp/pull/1995.

const int GGML_TASK_TYPE_INIT = 0;
const int GGML_TASK_TYPE_COMPUTE = 1;
const int GGML_TASK_TYPE_FINALIZE = 2;

const int GGML_TENSOR_FLAG_INPUT = 1;
const int GGML_TENSOR_FLAG_OUTPUT = 2;
const int GGML_TENSOR_FLAG_PARAM = 4;

/// NOTE: always add types at the end of the enum to keep backward compatibility

const int GGML_TYPE_F32 = 0;
const int GGML_TYPE_F16 = 1;
const int GGML_TYPE_Q4_0 = 2;
const int GGML_TYPE_Q4_1 = 3;

/// GGML_TYPE_Q4_2 = 4, support has been removed
/// GGML_TYPE_Q4_3 = 5, support has been removed
const int GGML_TYPE_Q5_0 = 6;
const int GGML_TYPE_Q5_1 = 7;
const int GGML_TYPE_Q8_0 = 8;
const int GGML_TYPE_Q8_1 = 9;
const int GGML_TYPE_Q2_K = 10;
const int GGML_TYPE_Q3_K = 11;
const int GGML_TYPE_Q4_K = 12;
const int GGML_TYPE_Q5_K = 13;
const int GGML_TYPE_Q6_K = 14;
const int GGML_TYPE_Q8_K = 15;
const int GGML_TYPE_IQ2_XXS = 16;
const int GGML_TYPE_IQ2_XS = 17;
const int GGML_TYPE_IQ3_XXS = 18;
const int GGML_TYPE_IQ1_S = 19;
const int GGML_TYPE_IQ4_NL = 20;
const int GGML_TYPE_IQ3_S = 21;
const int GGML_TYPE_IQ2_S = 22;
const int GGML_TYPE_IQ4_XS = 23;
const int GGML_TYPE_I8 = 24;
const int GGML_TYPE_I16 = 25;
const int GGML_TYPE_I32 = 26;
const int GGML_TYPE_I64 = 27;
const int GGML_TYPE_F64 = 28;
const int GGML_TYPE_IQ1_M = 29;
const int GGML_TYPE_COUNT = 30;

const int GGML_UNARY_OP_ABS = 0;
const int GGML_UNARY_OP_SGN = 1;
const int GGML_UNARY_OP_NEG = 2;
const int GGML_UNARY_OP_STEP = 3;
const int GGML_UNARY_OP_TANH = 4;
const int GGML_UNARY_OP_ELU = 5;
const int GGML_UNARY_OP_RELU = 6;
const int GGML_UNARY_OP_GELU = 7;
const int GGML_UNARY_OP_GELU_QUICK = 8;
const int GGML_UNARY_OP_SILU = 9;
const int GGML_UNARY_OP_HARDSWISH = 10;
const int GGML_UNARY_OP_HARDSIGMOID = 11;
const int GGML_UNARY_OP_COUNT = 12;

/// gguf

const int GGUF_TYPE_UINT8 = 0;
const int GGUF_TYPE_INT8 = 1;
const int GGUF_TYPE_UINT16 = 2;
const int GGUF_TYPE_INT16 = 3;
const int GGUF_TYPE_UINT32 = 4;
const int GGUF_TYPE_INT32 = 5;
const int GGUF_TYPE_FLOAT32 = 6;
const int GGUF_TYPE_BOOL = 7;
const int GGUF_TYPE_STRING = 8;
const int GGUF_TYPE_ARRAY = 9;
const int GGUF_TYPE_UINT64 = 10;
const int GGUF_TYPE_INT64 = 11;
const int GGUF_TYPE_FLOAT64 = 12;

/// marks the end of the enum
const int GGUF_TYPE_COUNT = 13;
const int GGML_DEFAULT_GRAPH_SIZE = 2048;
const int GGML_DEFAULT_N_THREADS = 4;
const int GGML_EXIT_ABORTED = 1;
const int GGML_EXIT_SUCCESS = 0;
const int GGML_FILE_MAGIC = 0x67676d6c; // "ggml"
const int GGML_FILE_VERSION = 1;
const int GGML_MAX_CONTEXTS = 64;
const int GGML_MAX_DIMS = 4;
const int GGML_MAX_NAME = 64;
const int GGML_MAX_OP_PARAMS = 64;
const int GGML_MAX_PARAMS = 2048;
const int GGML_MAX_SRC = 10;
const int GGML_MEM_ALIGN = 16;
const int GGML_N_TASKS_MAX = -1;
const int GGML_QNT_VERSION = 2;
const int GGML_QNT_VERSION_FACTOR = 1000;
const int GGUF_DEFAULT_ALIGNMENT = 32;
const String GGUF_MAGIC = 'GGUF';
const int GGUF_VERSION = 3;
