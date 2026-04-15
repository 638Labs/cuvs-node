#include "cagra.h"
#include "hnsw.h"
#include "resources.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <cuvs/neighbors/hnsw.h>

Napi::Object CagraIndex::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "CagraIndex", {
    StaticMethod("build", &CagraIndex::Build),
    StaticMethod("buildChunked", &CagraIndex::BuildChunked),
    InstanceMethod("search", &CagraIndex::Search),
    InstanceMethod("serialize", &CagraIndex::Serialize),
    StaticMethod("deserialize", &CagraIndex::Deserialize),
    InstanceMethod("toHnsw", &CagraIndex::ToHnsw),
  });

  Napi::FunctionReference* constructor = new Napi::FunctionReference();
  *constructor = Napi::Persistent(func);
  env.SetInstanceData(constructor);

  exports.Set("CagraIndex", func);
  return exports;
}

CagraIndex::CagraIndex(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<CagraIndex>(info), index_(0), owns_index_(false), d_dataset_(nullptr) {
}

CagraIndex::~CagraIndex() {
  if (owns_index_ && index_ != 0) {
    cuvsCagraIndexDestroy(index_);
  }
  if (d_dataset_ != nullptr) {
    cudaFree(d_dataset_);
  }
}

Napi::Value CagraIndex::Build(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3 || !info[0].IsObject() || !info[1].IsTypedArray() || !info[2].IsObject()) {
    Napi::TypeError::New(env, "Expected (Resources, Float32Array, { rows, cols })").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Resources* resources = Napi::ObjectWrap<Resources>::Unwrap(info[0].As<Napi::Object>());
  Napi::Float32Array dataset = info[1].As<Napi::Float32Array>();
  Napi::Object opts = info[2].As<Napi::Object>();

  int64_t rows = opts.Get("rows").As<Napi::Number>().Int64Value();
  int64_t cols = opts.Get("cols").As<Napi::Number>().Int64Value();

  if (rows <= 0 || cols <= 0) {
    Napi::RangeError::New(env, "rows and cols must be positive").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if ((size_t)(rows * cols) != dataset.ElementLength()) {
    Napi::RangeError::New(env, "dataset length does not match rows * cols").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  size_t length = 0;
  float* d_data = CopyToDevice(env, dataset, &length);
  if (d_data == nullptr) {
    return env.Undefined();
  }

  int64_t shape[2] = { rows, cols };
  DLManagedTensor tensor = {};
  tensor.dl_tensor.data = d_data;
  tensor.dl_tensor.device.device_type = kDLCUDA;
  tensor.dl_tensor.device.device_id = 0;
  tensor.dl_tensor.ndim = 2;
  tensor.dl_tensor.dtype.code = kDLFloat;
  tensor.dl_tensor.dtype.bits = 32;
  tensor.dl_tensor.dtype.lanes = 1;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = nullptr;
  tensor.dl_tensor.byte_offset = 0;

  cuvsCagraIndexParams_t params;
  if (cuvsCagraIndexParamsCreate(&params) != CUVS_SUCCESS) {
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsCagraIndexParamsCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsCagraIndex_t index;
  if (cuvsCagraIndexCreate(&index) != CUVS_SUCCESS) {
    cuvsCagraIndexParamsDestroy(params);
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsCagraIndexCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsError_t build_err = cuvsCagraBuild(resources->GetResource(), params, &tensor, index);
  cuvsCagraIndexParamsDestroy(params);

  if (build_err != CUVS_SUCCESS) {
    cuvsCagraIndexDestroy(index);
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsCagraBuild failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::FunctionReference* ctor = env.GetInstanceData<Napi::FunctionReference>();
  Napi::Object obj = ctor->New({});
  CagraIndex* wrapper = Napi::ObjectWrap<CagraIndex>::Unwrap(obj);
  wrapper->index_ = index;
  wrapper->owns_index_ = true;
  wrapper->d_dataset_ = d_data;
  return obj;
}

Napi::Value CagraIndex::BuildChunked(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3 || !info[0].IsObject() || !info[1].IsArray() || !info[2].IsObject()) {
    Napi::TypeError::New(env, "Expected (Resources, Float32Array[], { rows, cols })").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Resources* resources = Napi::ObjectWrap<Resources>::Unwrap(info[0].As<Napi::Object>());
  Napi::Array chunks = info[1].As<Napi::Array>();
  Napi::Object opts = info[2].As<Napi::Object>();

  int64_t rows = opts.Get("rows").As<Napi::Number>().Int64Value();
  int64_t cols = opts.Get("cols").As<Napi::Number>().Int64Value();

  if (rows <= 0 || cols <= 0) {
    Napi::RangeError::New(env, "rows and cols must be positive").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  size_t expected = (size_t)rows * (size_t)cols;
  float* d_data = CopyChunksToDevice(env, chunks, expected);
  if (d_data == nullptr) {
    return env.Undefined();
  }

  int64_t shape[2] = { rows, cols };
  DLManagedTensor tensor = {};
  tensor.dl_tensor.data = d_data;
  tensor.dl_tensor.device.device_type = kDLCUDA;
  tensor.dl_tensor.device.device_id = 0;
  tensor.dl_tensor.ndim = 2;
  tensor.dl_tensor.dtype.code = kDLFloat;
  tensor.dl_tensor.dtype.bits = 32;
  tensor.dl_tensor.dtype.lanes = 1;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = nullptr;
  tensor.dl_tensor.byte_offset = 0;

  cuvsCagraIndexParams_t params;
  if (cuvsCagraIndexParamsCreate(&params) != CUVS_SUCCESS) {
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsCagraIndexParamsCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsCagraIndex_t index;
  if (cuvsCagraIndexCreate(&index) != CUVS_SUCCESS) {
    cuvsCagraIndexParamsDestroy(params);
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsCagraIndexCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsError_t build_err = cuvsCagraBuild(resources->GetResource(), params, &tensor, index);
  cuvsCagraIndexParamsDestroy(params);

  if (build_err != CUVS_SUCCESS) {
    cuvsCagraIndexDestroy(index);
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsCagraBuild failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::FunctionReference* ctor = env.GetInstanceData<Napi::FunctionReference>();
  Napi::Object obj = ctor->New({});
  CagraIndex* wrapper = Napi::ObjectWrap<CagraIndex>::Unwrap(obj);
  wrapper->index_ = index;
  wrapper->owns_index_ = true;
  wrapper->d_dataset_ = d_data;
  return obj;
}

Napi::Value CagraIndex::Search(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3 || !info[0].IsObject() || !info[1].IsTypedArray() || !info[2].IsObject()) {
    Napi::TypeError::New(env, "Expected (Resources, Float32Array, { rows, cols, k })").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Resources* resources = Napi::ObjectWrap<Resources>::Unwrap(info[0].As<Napi::Object>());
  Napi::Float32Array queries = info[1].As<Napi::Float32Array>();
  Napi::Object opts = info[2].As<Napi::Object>();

  int64_t rows = opts.Get("rows").As<Napi::Number>().Int64Value();
  int64_t cols = opts.Get("cols").As<Napi::Number>().Int64Value();
  int64_t k = opts.Get("k").As<Napi::Number>().Int64Value();

  if (rows <= 0 || cols <= 0 || k <= 0) {
    Napi::RangeError::New(env, "rows, cols, k must be positive").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if ((size_t)(rows * cols) != queries.ElementLength()) {
    Napi::RangeError::New(env, "queries length does not match rows * cols").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  size_t qlen = 0;
  float* d_queries = CopyToDevice(env, queries, &qlen);
  if (d_queries == nullptr) return env.Undefined();

  size_t n_results = (size_t)rows * (size_t)k;
  uint32_t* d_neighbors = nullptr;
  float* d_distances = nullptr;
  if (cudaMalloc((void**)&d_neighbors, n_results * sizeof(uint32_t)) != cudaSuccess) {
    cudaFree(d_queries);
    Napi::Error::New(env, "cudaMalloc neighbors failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (cudaMalloc((void**)&d_distances, n_results * sizeof(float)) != cudaSuccess) {
    cudaFree(d_queries);
    cudaFree(d_neighbors);
    Napi::Error::New(env, "cudaMalloc distances failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  int64_t q_shape[2] = { rows, cols };
  int64_t r_shape[2] = { rows, k };

  DLManagedTensor q_tensor = {};
  q_tensor.dl_tensor.data = d_queries;
  q_tensor.dl_tensor.device.device_type = kDLCUDA;
  q_tensor.dl_tensor.device.device_id = 0;
  q_tensor.dl_tensor.ndim = 2;
  q_tensor.dl_tensor.dtype.code = kDLFloat;
  q_tensor.dl_tensor.dtype.bits = 32;
  q_tensor.dl_tensor.dtype.lanes = 1;
  q_tensor.dl_tensor.shape = q_shape;

  DLManagedTensor n_tensor = {};
  n_tensor.dl_tensor.data = d_neighbors;
  n_tensor.dl_tensor.device.device_type = kDLCUDA;
  n_tensor.dl_tensor.device.device_id = 0;
  n_tensor.dl_tensor.ndim = 2;
  n_tensor.dl_tensor.dtype.code = kDLUInt;
  n_tensor.dl_tensor.dtype.bits = 32;
  n_tensor.dl_tensor.dtype.lanes = 1;
  n_tensor.dl_tensor.shape = r_shape;

  DLManagedTensor d_tensor = {};
  d_tensor.dl_tensor.data = d_distances;
  d_tensor.dl_tensor.device.device_type = kDLCUDA;
  d_tensor.dl_tensor.device.device_id = 0;
  d_tensor.dl_tensor.ndim = 2;
  d_tensor.dl_tensor.dtype.code = kDLFloat;
  d_tensor.dl_tensor.dtype.bits = 32;
  d_tensor.dl_tensor.dtype.lanes = 1;
  d_tensor.dl_tensor.shape = r_shape;

  cuvsCagraSearchParams_t params;
  if (cuvsCagraSearchParamsCreate(&params) != CUVS_SUCCESS) {
    cudaFree(d_queries); cudaFree(d_neighbors); cudaFree(d_distances);
    Napi::Error::New(env, "cuvsCagraSearchParamsCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsFilter filter = { 0, NO_FILTER };
  cuvsError_t err = cuvsCagraSearch(resources->GetResource(), params, index_,
                                    &q_tensor, &n_tensor, &d_tensor, filter);
  cuvsCagraSearchParamsDestroy(params);

  if (err != CUVS_SUCCESS) {
    cudaFree(d_queries); cudaFree(d_neighbors); cudaFree(d_distances);
    Napi::Error::New(env, "cuvsCagraSearch failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::Uint32Array indices_out = CopyUint32FromDevice(env, d_neighbors, n_results);
  Napi::Float32Array distances_out = CopyFromDevice(env, d_distances, n_results);

  cudaFree(d_queries);
  cudaFree(d_neighbors);
  cudaFree(d_distances);

  Napi::Object result = Napi::Object::New(env);
  result.Set("indices", indices_out);
  result.Set("distances", distances_out);
  return result;
}

Napi::Value CagraIndex::Serialize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsString()) {
    Napi::TypeError::New(env, "Expected (Resources, string path)").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Resources* resources = Napi::ObjectWrap<Resources>::Unwrap(info[0].As<Napi::Object>());
  std::string path = info[1].As<Napi::String>().Utf8Value();

  if (index_ == 0) {
    Napi::Error::New(env, "index is not built").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsError_t err = cuvsCagraSerialize(resources->GetResource(), path.c_str(), index_, true);
  if (err != CUVS_SUCCESS) {
    Napi::Error::New(env, "cuvsCagraSerialize failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  return env.Undefined();
}

Napi::Value CagraIndex::Deserialize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsString()) {
    Napi::TypeError::New(env, "Expected (Resources, string path)").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Resources* resources = Napi::ObjectWrap<Resources>::Unwrap(info[0].As<Napi::Object>());
  std::string path = info[1].As<Napi::String>().Utf8Value();

  cuvsCagraIndex_t index;
  if (cuvsCagraIndexCreate(&index) != CUVS_SUCCESS) {
    Napi::Error::New(env, "cuvsCagraIndexCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsError_t err = cuvsCagraDeserialize(resources->GetResource(), path.c_str(), index);
  if (err != CUVS_SUCCESS) {
    cuvsCagraIndexDestroy(index);
    Napi::Error::New(env, "cuvsCagraDeserialize failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::FunctionReference* ctor = env.GetInstanceData<Napi::FunctionReference>();
  Napi::Object obj = ctor->New({});
  CagraIndex* wrapper = Napi::ObjectWrap<CagraIndex>::Unwrap(obj);
  wrapper->index_ = index;
  wrapper->owns_index_ = true;
  return obj;
}

Napi::Value CagraIndex::ToHnsw(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "Expected (Resources, opts?)").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Resources* resources = Napi::ObjectWrap<Resources>::Unwrap(info[0].As<Napi::Object>());

  if (index_ == 0) {
    Napi::Error::New(env, "cagra index is not built").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsHnswIndexParams_t params;
  if (cuvsHnswIndexParamsCreate(&params) != CUVS_SUCCESS) {
    Napi::Error::New(env, "cuvsHnswIndexParamsCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    if (opts.Has("hierarchy")) {
      params->hierarchy = (cuvsHnswHierarchy)opts.Get("hierarchy").As<Napi::Number>().Uint32Value();
    }
    if (opts.Has("efConstruction")) {
      params->ef_construction = opts.Get("efConstruction").As<Napi::Number>().Int32Value();
    }
    if (opts.Has("numThreads")) {
      params->num_threads = opts.Get("numThreads").As<Napi::Number>().Int32Value();
    }
  }

  cuvsHnswIndex_t hnsw_index;
  if (cuvsHnswIndexCreate(&hnsw_index) != CUVS_SUCCESS) {
    cuvsHnswIndexParamsDestroy(params);
    Napi::Error::New(env, "cuvsHnswIndexCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  hnsw_index->dtype.code = kDLFloat;
  hnsw_index->dtype.bits = 32;
  hnsw_index->dtype.lanes = 1;

  cuvsError_t err = cuvsHnswFromCagra(resources->GetResource(), params, index_, hnsw_index);
  cuvsHnswIndexParamsDestroy(params);

  if (err != CUVS_SUCCESS) {
    cuvsHnswIndexDestroy(hnsw_index);
    Napi::Error::New(env, "cuvsHnswFromCagra failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  return HnswIndex::WrapIndex(env, hnsw_index);
}
