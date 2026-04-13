#include <napi.h>
#include "resources.h"
#include "cagra.h"
#include "ivf_flat.h"
#include "ivf_pq.h"
#include "brute_force.h"
#include "hnsw.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  Resources::Init(env, exports);
  IvfFlatIndex::Init(env, exports);
  IvfPqIndex::Init(env, exports);
  BruteForceIndex::Init(env, exports);
  HnswIndex::Init(env, exports);
  CagraIndex::Init(env, exports);
  return exports;
}

NODE_API_MODULE(cuvs_node, Init)
