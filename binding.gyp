{
  "targets": [
    {
      "target_name": "cuvs_node",
      "sources": [
        "src/addon.cc",
        "src/resources.cc",
        "src/cagra.cc",
        "src/ivf_flat.cc",
        "src/ivf_pq.cc",
        "src/brute_force.cc",
        "src/hnsw.cc",
        "src/utils.cc"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "<!@(echo $CONDA_PREFIX/include)",
        "<!@(find /usr/local/cuda* -name 'cuda_runtime.h' -printf '%h' -quit 2>/dev/null || echo /usr/local/cuda/include)"
      ],
      "libraries": [
        "-L<!@(echo $CONDA_PREFIX/lib)",
        "-lcuvs_c",
        "-lcudart"
      ],
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "cflags_cc": ["-std=c++17"],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"]
    }
  ]
}
