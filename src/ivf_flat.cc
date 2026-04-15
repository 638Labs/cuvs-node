#include "ivf_flat.h"
#include "resources.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>

Napi::FunctionReference IvfFlatIndex::constructor_;

Napi::Object IvfFlatIndex::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "IvfFlatIndex", {
    StaticMethod("build", &IvfFlatIndex::Build),
    StaticMethod("buildChunked", &IvfFlatIndex::BuildChunked),
    InstanceMethod("search", &IvfFlatIndex::Search),
    InstanceMethod("serialize", &IvfFlatIndex::Serialize),
    StaticMethod("deserialize", &IvfFlatIndex::Deserialize),
  });

  constructor_ = Napi::Persistent(func);
  constructor_.SuppressDestruct();

  exports.Set("IvfFlatIndex", func);
  return exports;
}

IvfFlatIndex::IvfFlatIndex(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<IvfFlatIndex>(info), index_(0), owns_index_(false), d_dataset_(nullptr) {
}

IvfFlatIndex::~IvfFlatIndex() {
  if (owns_index_ && index_ != 0) {
    cuvsIvfFlatIndexDestroy(index_);
  }
  if (d_dataset_ != nullptr) {
    cudaFree(d_dataset_);
  }
}

Napi::Value IvfFlatIndex::Build(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3 || !info[0].IsObject() || !info[1].IsTypedArray() || !info[2].IsObject()) {
    Napi::TypeError::New(env, "Expected (Resources, Float32Array, { rows, cols, nLists? })").ThrowAsJavaScriptException();
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

  cuvsIvfFlatIndexParams_t params;
  if (cuvsIvfFlatIndexParamsCreate(&params) != CUVS_SUCCESS) {
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsIvfFlatIndexParamsCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  if (opts.Has("nLists")) {
    params->n_lists = opts.Get("nLists").As<Napi::Number>().Uint32Value();
  }

  cuvsIvfFlatIndex_t index;
  if (cuvsIvfFlatIndexCreate(&index) != CUVS_SUCCESS) {
    cuvsIvfFlatIndexParamsDestroy(params);
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsIvfFlatIndexCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsError_t build_err = cuvsIvfFlatBuild(resources->GetResource(), params, &tensor, index);
  cuvsIvfFlatIndexParamsDestroy(params);

  if (build_err != CUVS_SUCCESS) {
    cuvsIvfFlatIndexDestroy(index);
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsIvfFlatBuild failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::Object obj = constructor_.New({});
  IvfFlatIndex* wrapper = Napi::ObjectWrap<IvfFlatIndex>::Unwrap(obj);
  wrapper->index_ = index;
  wrapper->owns_index_ = true;
  wrapper->d_dataset_ = d_data;
  return obj;
}

Napi::Value IvfFlatIndex::BuildChunked(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3 || !info[0].IsObject() || !info[1].IsArray() || !info[2].IsObject()) {
    Napi::TypeError::New(env, "Expected (Resources, Float32Array[], { rows, cols, nLists? })").ThrowAsJavaScriptException();
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

  cuvsIvfFlatIndexParams_t params;
  if (cuvsIvfFlatIndexParamsCreate(&params) != CUVS_SUCCESS) {
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsIvfFlatIndexParamsCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  if (opts.Has("nLists")) {
    params->n_lists = opts.Get("nLists").As<Napi::Number>().Uint32Value();
  }

  cuvsIvfFlatIndex_t index;
  if (cuvsIvfFlatIndexCreate(&index) != CUVS_SUCCESS) {
    cuvsIvfFlatIndexParamsDestroy(params);
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsIvfFlatIndexCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsError_t build_err = cuvsIvfFlatBuild(resources->GetResource(), params, &tensor, index);
  cuvsIvfFlatIndexParamsDestroy(params);

  if (build_err != CUVS_SUCCESS) {
    cuvsIvfFlatIndexDestroy(index);
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsIvfFlatBuild failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::Object obj = constructor_.New({});
  IvfFlatIndex* wrapper = Napi::ObjectWrap<IvfFlatIndex>::Unwrap(obj);
  wrapper->index_ = index;
  wrapper->owns_index_ = true;
  wrapper->d_dataset_ = d_data;
  return obj;
}

Napi::Value IvfFlatIndex::Search(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3 || !info[0].IsObject() || !info[1].IsTypedArray() || !info[2].IsObject()) {
    Napi::TypeError::New(env, "Expected (Resources, Float32Array, { rows, cols, k, nProbes? })").ThrowAsJavaScriptException();
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
  int64_t* d_neighbors = nullptr;
  float* d_distances = nullptr;
  if (cudaMalloc((void**)&d_neighbors, n_results * sizeof(int64_t)) != cudaSuccess) {
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
  n_tensor.dl_tensor.dtype.code = kDLInt;
  n_tensor.dl_tensor.dtype.bits = 64;
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

  cuvsIvfFlatSearchParams_t params;
  if (cuvsIvfFlatSearchParamsCreate(&params) != CUVS_SUCCESS) {
    cudaFree(d_queries); cudaFree(d_neighbors); cudaFree(d_distances);
    Napi::Error::New(env, "cuvsIvfFlatSearchParamsCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  if (opts.Has("nProbes")) {
    params->n_probes = opts.Get("nProbes").As<Napi::Number>().Uint32Value();
  }

  cuvsFilter filter = { 0, NO_FILTER };
  cuvsError_t err = cuvsIvfFlatSearch(resources->GetResource(), params, index_,
                                      &q_tensor, &n_tensor, &d_tensor, filter);
  cuvsIvfFlatSearchParamsDestroy(params);

  if (err != CUVS_SUCCESS) {
    cudaFree(d_queries); cudaFree(d_neighbors); cudaFree(d_distances);
    Napi::Error::New(env, "cuvsIvfFlatSearch failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::BigInt64Array indices_out = Napi::BigInt64Array::New(env, n_results);
  if (cudaMemcpy(indices_out.Data(), d_neighbors, n_results * sizeof(int64_t),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(d_queries); cudaFree(d_neighbors); cudaFree(d_distances);
    Napi::Error::New(env, "cudaMemcpy neighbors D2H failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  Napi::Float32Array distances_out = CopyFromDevice(env, d_distances, n_results);

  cudaFree(d_queries);
  cudaFree(d_neighbors);
  cudaFree(d_distances);

  Napi::Object result = Napi::Object::New(env);
  result.Set("indices", indices_out);
  result.Set("distances", distances_out);
  return result;
}

Napi::Value IvfFlatIndex::Serialize(const Napi::CallbackInfo& info) {
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

  cuvsError_t err = cuvsIvfFlatSerialize(resources->GetResource(), path.c_str(), index_);
  if (err != CUVS_SUCCESS) {
    Napi::Error::New(env, "cuvsIvfFlatSerialize failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  return env.Undefined();
}

Napi::Value IvfFlatIndex::Deserialize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsString()) {
    Napi::TypeError::New(env, "Expected (Resources, string path)").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Resources* resources = Napi::ObjectWrap<Resources>::Unwrap(info[0].As<Napi::Object>());
  std::string path = info[1].As<Napi::String>().Utf8Value();

  cuvsIvfFlatIndex_t index;
  if (cuvsIvfFlatIndexCreate(&index) != CUVS_SUCCESS) {
    Napi::Error::New(env, "cuvsIvfFlatIndexCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsError_t err = cuvsIvfFlatDeserialize(resources->GetResource(), path.c_str(), index);
  if (err != CUVS_SUCCESS) {
    cuvsIvfFlatIndexDestroy(index);
    Napi::Error::New(env, "cuvsIvfFlatDeserialize failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::Object obj = constructor_.New({});
  IvfFlatIndex* wrapper = Napi::ObjectWrap<IvfFlatIndex>::Unwrap(obj);
  wrapper->index_ = index;
  wrapper->owns_index_ = true;
  return obj;
}
