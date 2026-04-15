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

float* CopyChunksToDevice(Napi::Env env, Napi::Array chunks, size_t expected_length) {
  uint32_t n_chunks = chunks.Length();
  if (n_chunks == 0) {
    Napi::RangeError::New(env, "chunks array is empty").ThrowAsJavaScriptException();
    return nullptr;
  }

  size_t bytes = expected_length * sizeof(float);
  float* d_ptr = nullptr;
  cudaError_t err = cudaMalloc((void**)&d_ptr, bytes);
  if (err != cudaSuccess) {
    Napi::Error::New(env, "cudaMalloc failed").ThrowAsJavaScriptException();
    return nullptr;
  }

  size_t offset = 0;
  for (uint32_t i = 0; i < n_chunks; i++) {
    Napi::Value v = chunks.Get(i);
    if (!v.IsTypedArray()) {
      cudaFree(d_ptr);
      Napi::TypeError::New(env, "each chunk must be a Float32Array").ThrowAsJavaScriptException();
      return nullptr;
    }
    Napi::TypedArray ta = v.As<Napi::TypedArray>();
    if (ta.TypedArrayType() != napi_float32_array) {
      cudaFree(d_ptr);
      Napi::TypeError::New(env, "each chunk must be a Float32Array").ThrowAsJavaScriptException();
      return nullptr;
    }
    Napi::Float32Array chunk = v.As<Napi::Float32Array>();
    size_t chunk_len = chunk.ElementLength();
    if (offset + chunk_len > expected_length) {
      cudaFree(d_ptr);
      Napi::RangeError::New(env, "chunks total length exceeds rows * cols").ThrowAsJavaScriptException();
      return nullptr;
    }
    if (chunk_len > 0) {
      err = cudaMemcpy(d_ptr + offset, chunk.Data(), chunk_len * sizeof(float),
                       cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        cudaFree(d_ptr);
        Napi::Error::New(env, "cudaMemcpy H2D failed").ThrowAsJavaScriptException();
        return nullptr;
      }
    }
    offset += chunk_len;
  }

  if (offset != expected_length) {
    cudaFree(d_ptr);
    Napi::RangeError::New(env, "chunks total length does not match rows * cols").ThrowAsJavaScriptException();
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
