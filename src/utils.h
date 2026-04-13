#ifndef CUVS_NODE_UTILS_H
#define CUVS_NODE_UTILS_H

#include <napi.h>
#include <cuda_runtime.h>

float* CopyToDevice(Napi::Env env, Napi::Float32Array arr, size_t* length);
Napi::Float32Array CopyFromDevice(Napi::Env env, float* d_ptr, size_t length);
Napi::Uint32Array CopyUint32FromDevice(Napi::Env env, uint32_t* d_ptr, size_t length);

#endif
