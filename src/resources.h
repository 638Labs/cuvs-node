#ifndef CUVS_NODE_RESOURCES_H
#define CUVS_NODE_RESOURCES_H

#include <napi.h>
#include <cuvs/core/c_api.h>

class Resources : public Napi::ObjectWrap<Resources> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  Resources(const Napi::CallbackInfo& info);
  ~Resources();

  cuvsResources_t GetResource() { return res_; }

 private:
  Napi::Value Dispose(const Napi::CallbackInfo& info);

  cuvsResources_t res_;
  bool disposed_;
};

#endif
