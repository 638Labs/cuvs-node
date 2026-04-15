#ifndef CUVS_NODE_IVF_PQ_H
#define CUVS_NODE_IVF_PQ_H

#include <napi.h>
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_pq.h>

class IvfPqIndex : public Napi::ObjectWrap<IvfPqIndex> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  IvfPqIndex(const Napi::CallbackInfo& info);
  ~IvfPqIndex();

 private:
  static Napi::Value Build(const Napi::CallbackInfo& info);
  static Napi::Value BuildChunked(const Napi::CallbackInfo& info);
  Napi::Value Search(const Napi::CallbackInfo& info);
  Napi::Value Serialize(const Napi::CallbackInfo& info);
  static Napi::Value Deserialize(const Napi::CallbackInfo& info);

  static Napi::FunctionReference constructor_;

  cuvsIvfPqIndex_t index_;
  bool owns_index_;
  float* d_dataset_;
};

#endif
