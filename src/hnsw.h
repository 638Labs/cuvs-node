#ifndef CUVS_NODE_HNSW_H
#define CUVS_NODE_HNSW_H

#include <napi.h>
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/hnsw.h>

class HnswIndex : public Napi::ObjectWrap<HnswIndex> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  HnswIndex(const Napi::CallbackInfo& info);
  ~HnswIndex();

  static Napi::Object WrapIndex(Napi::Env env, cuvsHnswIndex_t idx);

 private:
  Napi::Value SearchImpl(const Napi::CallbackInfo& info);
  Napi::Value Serialize(const Napi::CallbackInfo& info);
  static Napi::Value Deserialize(const Napi::CallbackInfo& info);

  static Napi::FunctionReference constructor_;

  cuvsHnswIndex_t index_;
  bool owns_index_;
};

#endif
