#include "brute_force.h"
#include "resources.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>

Napi::FunctionReference BruteForceIndex::constructor_;

Napi::Object BruteForceIndex::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "BruteForceIndex", {
    StaticMethod("build", &BruteForceIndex::Build),
    InstanceMethod("search", &BruteForceIndex::Search),
    InstanceMethod("serialize", &BruteForceIndex::Serialize),
    StaticMethod("deserialize", &BruteForceIndex::Deserialize),
  });

  constructor_ = Napi::Persistent(func);
  constructor_.SuppressDestruct();

  exports.Set("BruteForceIndex", func);
  return exports;
}

BruteForceIndex::BruteForceIndex(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<BruteForceIndex>(info), index_(0), owns_index_(false), d_dataset_(nullptr) {
}

BruteForceIndex::~BruteForceIndex() {
  if (owns_index_ && index_ != 0) {
    cuvsBruteForceIndexDestroy(index_);
  }
  if (d_dataset_ != nullptr) {
    cudaFree(d_dataset_);
  }
}

Napi::Value BruteForceIndex::Build(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3 || !info[0].IsObject() || !info[1].IsTypedArray() || !info[2].IsObject()) {
    Napi::TypeError::New(env, "Expected (Resources, Float32Array, { rows, cols, metric? })").ThrowAsJavaScriptException();
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

  cuvsDistanceType metric = L2Expanded;
  if (opts.Has("metric")) {
    metric = (cuvsDistanceType)opts.Get("metric").As<Napi::Number>().Uint32Value();
  }
  float metric_arg = 0.0f;
  if (opts.Has("metricArg")) {
    metric_arg = opts.Get("metricArg").As<Napi::Number>().FloatValue();
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

  cuvsBruteForceIndex_t index;
  if (cuvsBruteForceIndexCreate(&index) != CUVS_SUCCESS) {
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsBruteForceIndexCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsError_t build_err = cuvsBruteForceBuild(resources->GetResource(), &tensor, metric, metric_arg, index);

  if (build_err != CUVS_SUCCESS) {
    cuvsBruteForceIndexDestroy(index);
    cudaFree(d_data);
    Napi::Error::New(env, "cuvsBruteForceBuild failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::Object obj = constructor_.New({});
  BruteForceIndex* wrapper = Napi::ObjectWrap<BruteForceIndex>::Unwrap(obj);
  wrapper->index_ = index;
  wrapper->owns_index_ = true;
  wrapper->d_dataset_ = d_data;
  return obj;
}

Napi::Value BruteForceIndex::Search(const Napi::CallbackInfo& info) {
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

  cuvsFilter filter = { 0, NO_FILTER };
  cuvsError_t err = cuvsBruteForceSearch(resources->GetResource(), index_,
                                         &q_tensor, &n_tensor, &d_tensor, filter);

  if (err != CUVS_SUCCESS) {
    cudaFree(d_queries); cudaFree(d_neighbors); cudaFree(d_distances);
    Napi::Error::New(env, "cuvsBruteForceSearch failed").ThrowAsJavaScriptException();
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

Napi::Value BruteForceIndex::Serialize(const Napi::CallbackInfo& info) {
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

  cuvsError_t err = cuvsBruteForceSerialize(resources->GetResource(), path.c_str(), index_);
  if (err != CUVS_SUCCESS) {
    Napi::Error::New(env, "cuvsBruteForceSerialize failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  return env.Undefined();
}

Napi::Value BruteForceIndex::Deserialize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsString()) {
    Napi::TypeError::New(env, "Expected (Resources, string path)").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Resources* resources = Napi::ObjectWrap<Resources>::Unwrap(info[0].As<Napi::Object>());
  std::string path = info[1].As<Napi::String>().Utf8Value();

  cuvsBruteForceIndex_t index;
  if (cuvsBruteForceIndexCreate(&index) != CUVS_SUCCESS) {
    Napi::Error::New(env, "cuvsBruteForceIndexCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  cuvsError_t err = cuvsBruteForceDeserialize(resources->GetResource(), path.c_str(), index);
  if (err != CUVS_SUCCESS) {
    cuvsBruteForceIndexDestroy(index);
    Napi::Error::New(env, "cuvsBruteForceDeserialize failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::Object obj = constructor_.New({});
  BruteForceIndex* wrapper = Napi::ObjectWrap<BruteForceIndex>::Unwrap(obj);
  wrapper->index_ = index;
  wrapper->owns_index_ = true;
  return obj;
}
