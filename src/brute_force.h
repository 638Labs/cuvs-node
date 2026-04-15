#ifndef CUVS_NODE_BRUTE_FORCE_H
#define CUVS_NODE_BRUTE_FORCE_H

#include <napi.h>
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/brute_force.h>

class BruteForceIndex : public Napi::ObjectWrap<BruteForceIndex> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  BruteForceIndex(const Napi::CallbackInfo& info);
  ~BruteForceIndex();

 private:
  static Napi::Value Build(const Napi::CallbackInfo& info);
  static Napi::Value BuildChunked(const Napi::CallbackInfo& info);
  Napi::Value Search(const Napi::CallbackInfo& info);
  Napi::Value Serialize(const Napi::CallbackInfo& info);
  static Napi::Value Deserialize(const Napi::CallbackInfo& info);

  static Napi::FunctionReference constructor_;

  cuvsBruteForceIndex_t index_;
  bool owns_index_;
  float* d_dataset_;
};

#endif
