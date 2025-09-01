/**
 * Common Header File for Tensor Parallel Implementations
 * 
 * This header contains shared data structures, utility functions, and configurations
 * used across all tensor parallel MOE and attention implementations.
 */

#ifndef COMMON_H
#define COMMON_H

#include <chrono>
#include <iostream>
#include <cassert>
#include <random>
#include <algorithm>
#include <utility>
#include <cmath>
#include <cstring>
#include <thread>
#include <fstream>
#include <string>
#include <vector>

/**
 * Configuration for the MOE transformer block
 * Defines model dimensions and expert configuration
 */
struct QwenConfig {
  int d_model = 2048;    // Model hidden dimension
  int d_ff = 768;        // Feed-forward dimension (per expert)
  int n_experts = 128;   // Total number of experts
  int top_k = 8;         // Number of experts to activate per token
};

/**
 * Configuration for attention layers
 * Defines attention-specific parameters
 */
struct AttnConfig {
  int d_model = 2048;    // Model hidden dimension
  int n_q = 32;          // Number of query heads
  int n_kv = 4;          // Number of key-value heads (for GQA)
  int head_dim = 64;     // Dimension per attention head (d_model / n_q)
  int max_seq_len = 2048; // Maximum sequence length for positional embeddings
};

/**
 * MOE layer weights structure (simplified, no biases)
 * Contains all the weight matrices needed for the MOE computation
 */
struct QwenLayerWeights {
  float* router_w = nullptr;  // Router weights [n_experts × d_model] - routes tokens to experts
  float** Wg = nullptr;       // Gate projection weights per expert [d_model × d_ff]
  float** Wu = nullptr;       // Up projection weights per expert [d_model × d_ff]  
  float** Wd = nullptr;       // Down projection weights per expert [d_ff × d_model]
};

/**
 * 2D tensor representation with row-major storage
 * Used to represent weight matrix shards in tensor parallel execution
 */
struct Tensor2D {
  float* data;  // Flat data pointer to weight values
  int rows;     // Number of rows in the tensor
  int cols;     // Number of columns in the tensor

  // Constructor
  Tensor2D(float* d = nullptr, int r = 0, int c = 0) : data(d), rows(r), cols(c) {}

  // Get total number of elements
  size_t size() const { return static_cast<size_t>(rows) * cols; }

  // Access element at position (i,j)
  float& at(int i, int j) { return data[i * cols + j]; }
  const float& at(int i, int j) const { return data[i * cols + j]; }
};

/**
 * Performance profiler for timing different components of computation
 * Tracks time spent in each phase of the forward pass
 */
struct Profiler {
  double router_time = 0.0;      // Time spent in router computation
  double topk_time = 0.0;        // Time spent in top-k expert selection
  double gate_proj_time = 0.0;   // Time spent in gate projections
  double up_proj_time = 0.0;     // Time spent in up projections
  double silu_mul_time = 0.0;    // Time spent in SiLU activation and multiplication
  double down_proj_time = 0.0;   // Time spent in down projections
  double comm_time = 0.0;        // Time spent in communication (MPI reduce operations)
  double total_time = 0.0;       // Total computation time

  // Attention-specific timing fields
  double rms_time = 0.0;         // Time spent in RMS normalization
  double qkv_proj_time = 0.0;    // Time spent in QKV projections
  double qk_norm_time = 0.0;     // Time spent in Q/K normalization
  double rope_time = 0.0;        // Time spent in RoPE (rotary position embedding)
  double scores_time = 0.0;      // Time spent computing attention scores
  double softmax_time = 0.0;     // Time spent in softmax
  double weighted_v_time = 0.0;  // Time spent in weighted value computation
  double o_proj_time = 0.0;      // Time spent in output projection

  // Print detailed timing breakdown
  void print(const std::string& prefix) const {
    std::cout << prefix << " Breakdown:" << std::endl;
    
    // MOE timings
    if (router_time > 0 || gate_proj_time > 0) {
      std::cout << "  Router: " << router_time << "s" << std::endl;
      std::cout << "  TopK: " << topk_time << "s" << std::endl;
      std::cout << "  Gate Proj: " << gate_proj_time << "s" << std::endl;
      std::cout << "  Up Proj: " << up_proj_time << "s" << std::endl;
      std::cout << "  SiLU + Mul: " << silu_mul_time << "s" << std::endl;
      std::cout << "  Down Proj: " << down_proj_time << "s" << std::endl;
    }
    
    // Attention timings
    if (rms_time > 0 || qkv_proj_time > 0) {
      std::cout << "  RMS: " << rms_time << "s" << std::endl;
      std::cout << "  QKV Proj: " << qkv_proj_time << "s" << std::endl;
      std::cout << "  Q/K Norm: " << qk_norm_time << "s" << std::endl;
      std::cout << "  RoPE: " << rope_time << "s" << std::endl;
      std::cout << "  Scores: " << scores_time << "s" << std::endl;
      std::cout << "  Softmax: " << softmax_time << "s" << std::endl;
      std::cout << "  Weighted V: " << weighted_v_time << "s" << std::endl;
      std::cout << "  O Proj: " << o_proj_time << "s" << std::endl;
    }
    
    std::cout << "  Comm: " << comm_time << "s" << std::endl;
    std::cout << "  Total: " << total_time << "s" << std::endl;
  }
};

// ============================================================================
// Common Utility Functions
// ============================================================================

/**
 * Basic matrix multiplication: C = A * B
 * @param A Input matrix A [m × k]
 * @param B Input matrix B [k × n] 
 * @param C Output matrix C [m × n]
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
inline void serial_matmul(const float* A, const float* B, float* C, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int p = 0; p < k; ++p) {
        sum += A[i * k + p] * B[p * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

/**
 * SiLU (Swish) activation function applied in-place: x = x / (1 + exp(-x))
 * @param x Input/output array to apply activation to
 * @param n Number of elements in the array
 */
inline void silu(float* x, int n) {
  for (int i = 0; i < n; ++i) {
    x[i] = x[i] / (1.0f + expf(-x[i]));
  }
}

/**
 * RMS normalization: out = x / sqrt(mean(x^2) + eps) * weight
 * @param out Output normalized vector [d_model]
 * @param x Input vector [d_model] 
 * @param weight RMS weight vector [d_model]
 * @param d_model Dimension of the vectors
 * @param eps Small epsilon for numerical stability
 */
inline void rms_norm(float* out, const float* x, const float* weight, int d_model, float eps = 1e-6f) {
  float sum_sq = 0.0f;
  for (int i = 0; i < d_model; ++i) {
    sum_sq += x[i] * x[i];
  }
  float rms = sqrtf(sum_sq / d_model + eps);
  for (int i = 0; i < d_model; ++i) {
    out[i] = (x[i] / rms) * weight[i];
  }
}

/**
 * Top-k expert selection with softmax normalization
 * Selects the k experts with highest routing logits and normalizes their weights
 * @param logits Router logits [n_experts] (modified in-place for top-k)
 * @param indices Output array for selected expert indices [k]
 * @param k Number of experts to select
 * @param n Total number of experts
 * @param scratch Temporary buffer [n_experts] for computation
 */
inline void topk(float* logits, int* indices, int k, int n, float* scratch) {
  // Copy logits to scratch to preserve original values during sorting
  memcpy(scratch, logits, n * sizeof(float));

  // Pair values with indices for sorting
  std::vector<std::pair<float, int>> pairs(n);
  for (int i = 0; i < n; ++i) {
    pairs[i] = {scratch[i], i};
  }

  // Partial sort to get top-k (descending order)
  std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                   [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                     return a.first > b.first;  // Descending order
                   });

  // Extract top-k indices and values
  for (int i = 0; i < k; ++i) {
    indices[i] = pairs[i].second;
    logits[i] = pairs[i].first;
  }

  // Apply softmax normalization to top-k values
  float max_logit = logits[0];  // Already sorted, so first is max
  for (int i = 1; i < k; ++i) {
    if (logits[i] > max_logit) max_logit = logits[i];
  }
  
  float sum_exp = 0.0f;
  for (int i = 0; i < k; ++i) {
    logits[i] = expf(logits[i] - max_logit);
    sum_exp += logits[i];
  }
  
  for (int i = 0; i < k; ++i) {
    logits[i] /= sum_exp;  // Normalize to sum to 1
  }
}

/**
 * Parse command line arguments for tensor parallel tests
 * @param argc Number of command line arguments
 * @param argv Command line argument strings
 * @param tp Output: tensor parallelism degree
 * @param batch Output: batch size
 * @return true if parsing successful, false otherwise
 */
inline bool parse_args(int argc, char* argv[], int& tp, int& batch) {
  tp = 1;      // Default tensor parallelism
  batch = 1;   // Default batch size
  
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--tp" && i + 1 < argc) {
      tp = std::atoi(argv[i + 1]);
      i++;
    } else if (std::string(argv[i]) == "--batch" && i + 1 < argc) {
      batch = std::atoi(argv[i + 1]);
      i++;
    } else {
      std::cerr << "Usage: " << argv[0] << " [--tp <tensor_parallelism>] [--batch <batch_size>]" << std::endl;
      return false;
    }
  }
  
  return true;
}

#endif // COMMON_H