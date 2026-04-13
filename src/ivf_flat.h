#ifndef CUVS_NODE_IVF_FLAT_H
#define CUVS_NODE_IVF_FLAT_H

#include <napi.h>
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/ivf_flat.h>

class IvfFlatIndex : public Napi::ObjectWrap<IvfFlatIndex> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  IvfFlatIndex(const Napi::CallbackInfo& info);
  ~IvfFlatIndex();

 private:
  static Napi::Value Build(const Napi::CallbackInfo& info);
  Napi::Value Search(const Napi::CallbackInfo& info);
  Napi::Value Serialize(const Napi::CallbackInfo& info);
  static Napi::Value Deserialize(const Napi::CallbackInfo& info);

  static Napi::FunctionReference constructor_;

  cuvsIvfFlatIndex_t index_;
  bool owns_index_;
  float* d_dataset_;
};

#endif
