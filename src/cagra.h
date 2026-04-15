#ifndef CUVS_NODE_CAGRA_H
#define CUVS_NODE_CAGRA_H

#include <napi.h>
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>

class CagraIndex : public Napi::ObjectWrap<CagraIndex> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  CagraIndex(const Napi::CallbackInfo& info);
  ~CagraIndex();

 private:
  static Napi::Value Build(const Napi::CallbackInfo& info);
  static Napi::Value BuildChunked(const Napi::CallbackInfo& info);
  Napi::Value Search(const Napi::CallbackInfo& info);
  Napi::Value Serialize(const Napi::CallbackInfo& info);
  static Napi::Value Deserialize(const Napi::CallbackInfo& info);
  Napi::Value ToHnsw(const Napi::CallbackInfo& info);

  cuvsCagraIndex_t index_;
  bool owns_index_;
  float* d_dataset_;
};

#endif
