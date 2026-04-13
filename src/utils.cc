#include "utils.h"

float* CopyToDevice(Napi::Env env, Napi::Float32Array arr, size_t* length) {
  *length = arr.ElementLength();
  size_t bytes = *length * sizeof(float);

  float* d_ptr = nullptr;
  cudaError_t err = cudaMalloc((void**)&d_ptr, bytes);
  if (err != cudaSuccess) {
    Napi::Error::New(env, "cudaMalloc failed").ThrowAsJavaScriptException();
    return nullptr;
  }

  err = cudaMemcpy(d_ptr, arr.Data(), bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_ptr);
    Napi::Error::New(env, "cudaMemcpy H2D failed").ThrowAsJavaScriptException();
    return nullptr;
  }

  return d_ptr;
}

Napi::Float32Array CopyFromDevice(Napi::Env env, float* d_ptr, size_t length) {
  Napi::Float32Array arr = Napi::Float32Array::New(env, length);
  size_t bytes = length * sizeof(float);

  cudaError_t err = cudaMemcpy(arr.Data(), d_ptr, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    Napi::Error::New(env, "cudaMemcpy D2H failed").ThrowAsJavaScriptException();
  }

  return arr;
}

Napi::Uint32Array CopyUint32FromDevice(Napi::Env env, uint32_t* d_ptr, size_t length) {
  Napi::Uint32Array arr = Napi::Uint32Array::New(env, length);
  size_t bytes = length * sizeof(uint32_t);

  cudaError_t err = cudaMemcpy(arr.Data(), d_ptr, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    Napi::Error::New(env, "cudaMemcpy D2H uint32 failed").ThrowAsJavaScriptException();
  }

  return arr;
}
