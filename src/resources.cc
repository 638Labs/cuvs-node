#include "resources.h"

Napi::Object Resources::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "Resources", {
    InstanceMethod("dispose", &Resources::Dispose),
  });

  Napi::FunctionReference* constructor = new Napi::FunctionReference();
  *constructor = Napi::Persistent(func);
  env.SetInstanceData(constructor);

  exports.Set("Resources", func);
  return exports;
}

Resources::Resources(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Resources>(info), res_(0), disposed_(false) {
  Napi::Env env = info.Env();

  cuvsError_t err = cuvsResourcesCreate(&res_);
  if (err != CUVS_SUCCESS) {
    Napi::Error::New(env, "Failed to create cuVS resources").ThrowAsJavaScriptException();
    return;
  }
}

Resources::~Resources() {
  if (!disposed_ && res_ != 0) {
    cuvsResourcesDestroy(res_);
  }
}

Napi::Value Resources::Dispose(const Napi::CallbackInfo& info) {
  if (!disposed_ && res_ != 0) {
    cuvsResourcesDestroy(res_);
    disposed_ = true;
  }
  return info.Env().Undefined();
}
