#ifndef CUVS_NODE_UTILS_H
#define CUVS_NODE_UTILS_H

#include <napi.h>
#include <cuda_runtime.h>
#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>

float* CopyToDevice(Napi::Env env, Napi::Float32Array arr, size_t* length);
float* CopyChunksToDevice(Napi::Env env, Napi::Array chunks, size_t expected_length);
Napi::Float32Array CopyFromDevice(Napi::Env env, float* d_ptr, size_t length);
Napi::Uint32Array CopyUint32FromDevice(Napi::Env env, uint32_t* d_ptr, size_t length);

// Reads opts.metric (string) and returns the matching cuvsDistanceType.
// Defaults to L2Expanded if absent. Throws on unknown strings or non-string values.
cuvsDistanceType ResolveMetric(Napi::Env env, Napi::Object opts);

#endif
