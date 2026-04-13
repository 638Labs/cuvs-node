#include "hnsw.h"
#include "resources.h"
#include <dlpack/dlpack.h>
#include <vector>

Napi::FunctionReference HnswIndex::constructor_;

Napi::Object HnswIndex::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "HnswIndex", {
    InstanceMethod("search", &HnswIndex::SearchImpl),
    InstanceMethod("serialize", &HnswIndex::Serialize),
    StaticMethod("deserialize", &HnswIndex::Deserialize),
  });

  constructor_ = Napi::Persistent(func);
  constructor_.SuppressDestruct();

  exports.Set("HnswIndex", func);
  return exports;
}

HnswIndex::HnswIndex(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<HnswIndex>(info), index_(0), owns_index_(false) {
}

HnswIndex::~HnswIndex() {
  if (owns_index_ && index_ != 0) {
    cuvsHnswIndexDestroy(index_);
  }
}

Napi::Object HnswIndex::WrapIndex(Napi::Env env, cuvsHnswIndex_t idx) {
  Napi::Object obj = constructor_.New({});
  HnswIndex* wrapper = Napi::ObjectWrap<HnswIndex>::Unwrap(obj);
  wrapper->index_ = idx;
  wrapper->owns_index_ = true;
  return obj;
}

Napi::Value HnswIndex::SearchImpl(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3 || !info[0].IsObject() || !info[1].IsTypedArray() || !info[2].IsObject()) {
    Napi::TypeError::New(env, "Expected (Resources, Float32Array, { rows, cols, k, ef? })").ThrowAsJavaScriptException();
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

  size_t n_results = (size_t)rows * (size_t)k;
  std::vector<uint64_t> h_neighbors(n_results);
  std::vector<float> h_distances(n_results);

  int64_t q_shape[2] = { rows, cols };
  int64_t r_shape[2] = { rows, k };

  DLManagedTensor q_tensor = {};
  q_tensor.dl_tensor.data = (void*)queries.Data();
  q_tensor.dl_tensor.device.device_type = kDLCPU;
  q_tensor.dl_tensor.device.device_id = 0;
  q_tensor.dl_tensor.ndim = 2;
  q_tensor.dl_tensor.dtype.code = kDLFloat;
  q_tensor.dl_tensor.dtype.bits = 32;
  q_tensor.dl_tensor.dtype.lanes = 1;
  q_tensor.dl_tensor.shape = q_shape;

  DLManagedTensor n_tensor = {};
  n_tensor.dl_tensor.data = h_neighbors.data();
  n_tensor.dl_tensor.device.device_type = kDLCPU;
  n_tensor.dl_tensor.device.device_id = 0;
  n_tensor.dl_tensor.ndim = 2;
  n_tensor.dl_tensor.dtype.code = kDLUInt;
  n_tensor.dl_tensor.dtype.bits = 64;
  n_tensor.dl_tensor.dtype.lanes = 1;
  n_tensor.dl_tensor.shape = r_shape;

  DLManagedTensor d_tensor = {};
  d_tensor.dl_tensor.data = h_distances.data();
  d_tensor.dl_tensor.device.device_type = kDLCPU;
  d_tensor.dl_tensor.device.device_id = 0;
  d_tensor.dl_tensor.ndim = 2;
  d_tensor.dl_tensor.dtype.code = kDLFloat;
  d_tensor.dl_tensor.dtype.bits = 32;
  d_tensor.dl_tensor.dtype.lanes = 1;
  d_tensor.dl_tensor.shape = r_shape;

  cuvsHnswSearchParams_t params;
  if (cuvsHnswSearchParamsCreate(&params) != CUVS_SUCCESS) {
    Napi::Error::New(env, "cuvsHnswSearchParamsCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (opts.Has("ef")) {
    params->ef = opts.Get("ef").As<Napi::Number>().Int32Value();
  }
  if (opts.Has("numThreads")) {
    params->num_threads = opts.Get("numThreads").As<Napi::Number>().Int32Value();
  }

  cuvsError_t err = cuvsHnswSearch(resources->GetResource(), params, index_,
                                   &q_tensor, &n_tensor, &d_tensor);
  cuvsHnswSearchParamsDestroy(params);

  if (err != CUVS_SUCCESS) {
    Napi::Error::New(env, "cuvsHnswSearch failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::BigUint64Array indices_out = Napi::BigUint64Array::New(env, n_results);
  std::memcpy(indices_out.Data(), h_neighbors.data(), n_results * sizeof(uint64_t));
  Napi::Float32Array distances_out = Napi::Float32Array::New(env, n_results);
  std::memcpy(distances_out.Data(), h_distances.data(), n_results * sizeof(float));

  Napi::Object result = Napi::Object::New(env);
  result.Set("indices", indices_out);
  result.Set("distances", distances_out);
  return result;
}

Napi::Value HnswIndex::Serialize(const Napi::CallbackInfo& info) {
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

  cuvsError_t err = cuvsHnswSerialize(resources->GetResource(), path.c_str(), index_);
  if (err != CUVS_SUCCESS) {
    Napi::Error::New(env, "cuvsHnswSerialize failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  return env.Undefined();
}

Napi::Value HnswIndex::Deserialize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3 || !info[0].IsObject() || !info[1].IsString() || !info[2].IsObject()) {
    Napi::TypeError::New(env, "Expected (Resources, string path, { dim, metric? })").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Resources* resources = Napi::ObjectWrap<Resources>::Unwrap(info[0].As<Napi::Object>());
  std::string path = info[1].As<Napi::String>().Utf8Value();
  Napi::Object opts = info[2].As<Napi::Object>();

  if (!opts.Has("dim")) {
    Napi::TypeError::New(env, "dim is required").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  int dim = opts.Get("dim").As<Napi::Number>().Int32Value();
  cuvsDistanceType metric = L2Expanded;
  if (opts.Has("metric")) {
    metric = (cuvsDistanceType)opts.Get("metric").As<Napi::Number>().Uint32Value();
  }

  cuvsHnswIndexParams_t params;
  if (cuvsHnswIndexParamsCreate(&params) != CUVS_SUCCESS) {
    Napi::Error::New(env, "cuvsHnswIndexParamsCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (opts.Has("hierarchy")) {
    params->hierarchy = (cuvsHnswHierarchy)opts.Get("hierarchy").As<Napi::Number>().Uint32Value();
  }

  cuvsHnswIndex_t index;
  if (cuvsHnswIndexCreate(&index) != CUVS_SUCCESS) {
    cuvsHnswIndexParamsDestroy(params);
    Napi::Error::New(env, "cuvsHnswIndexCreate failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  index->dtype.code = kDLFloat;
  index->dtype.bits = 32;
  index->dtype.lanes = 1;

  cuvsError_t err = cuvsHnswDeserialize(resources->GetResource(), params, path.c_str(),
                                        dim, metric, index);
  cuvsHnswIndexParamsDestroy(params);

  if (err != CUVS_SUCCESS) {
    cuvsHnswIndexDestroy(index);
    Napi::Error::New(env, "cuvsHnswDeserialize failed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  return WrapIndex(env, index);
}
