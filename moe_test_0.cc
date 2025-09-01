/**
 * MOE Test 0: Tensor Parallel Mixture of Experts Implementation
 * 
 * This file implements a tensor parallel (TP) version of a Mixture of Experts (MOE) block
 * without MPI support. It uses local threading for parallelism and processes single tokens.
 * 
 * Key Features:
 * - Tensor parallelism using local threads (no MPI)
 * - Single token processing (batch_size = 1)
 * - MOE with top-k expert selection
 * - Performance profiling and correctness validation
 */

#include <chrono>
#include <iostream>
#include <cassert>
#include <random>
#include <algorithm>
#include <utility>
#include <cmath>
#include <cstring>
#include <thread>

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
 * Performance profiler for timing different components of MOE computation
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

  // Print detailed timing breakdown
  void print(const std::string& prefix) const {
    std::cout << prefix << " Breakdown:" << std::endl;
    std::cout << "  Router: " << router_time << "s" << std::endl;
    std::cout << "  TopK: " << topk_time << "s" << std::endl;
    std::cout << "  Gate Proj: " << gate_proj_time << "s" << std::endl;
    std::cout << "  Up Proj: " << up_proj_time << "s" << std::endl;
    std::cout << "  SiLU + Mul: " << silu_mul_time << "s" << std::endl;
    std::cout << "  Down Proj: " << down_proj_time << "s" << std::endl;
    std::cout << "  Comm (Reduce): " << comm_time << "s" << std::endl;
    std::cout << "  Total: " << total_time << "s" << std::endl;
  }
};

/**
 * Basic matrix multiplication: C = A * B
 * @param A Input matrix A [m × k]
 * @param B Input matrix B [k × n] 
 * @param C Output matrix C [m × n]
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
void serial_matmul(const float* A, const float* B, float* C, int m, int n, int k) {
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
void silu(float* x, int n) {
  for (int i = 0; i < n; ++i) {
    x[i] = x[i] / (1.0f + expf(-x[i]));
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
void topk(float* logits, int* indices, int k, int n, float* scratch) {
  // Copy logits to scratch to preserve original values during sorting
  memcpy(scratch, logits, n * sizeof(float));

  // Pair values with indices
  std::vector<std::pair<float, int>> pairs(n);
  for (int i = 0; i < n; ++i) {
    pairs[i] = {scratch[i], i};
  }

  // Partial sort to get top-k (descending)
  std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

  // Softmax on top-k
  float max_log = pairs[0].first;
  float sum_exp = 0.0f;
  for (int i = 0; i < k; ++i) {
    float val = expf(pairs[i].first - max_log);
    sum_exp += val;
    logits[i] = val;
  }
  for (int i = 0; i < k; ++i) {
    logits[i] /= sum_exp;
    indices[i] = pairs[i].second;
  }
}

// Serial MoE layer
void moe_layer_serial(float* out, const float* x, const QwenLayerWeights* layer_weights,
                      const QwenConfig* cfg, int batch_size, Profiler& prof) {
  assert(batch_size == 1);  // For prototype
  int d_model = cfg->d_model;
  int d_ff = cfg->d_ff;
  int n_experts = cfg->n_experts;
  int top_k = cfg->top_k;

  // Local buffers (heap-allocated)
  float* router_logits = new float[n_experts];
  int* selected_experts = new int[top_k];
  float* expert_weights = new float[top_k];
  float* topk_scratch = new float[n_experts];
  float* gate_out = new float[d_ff];
  float* up_out = new float[d_ff];
  float* expert_out = new float[d_model];

  // Clear output
  memset(out, 0, d_model * sizeof(float));

  auto start_total = std::chrono::high_resolution_clock::now();

  // Router computation
  auto start = std::chrono::high_resolution_clock::now();
  serial_matmul(x, layer_weights->router_w, router_logits, 1, n_experts, d_model);
  auto end = std::chrono::high_resolution_clock::now();
  prof.router_time += std::chrono::duration<double>(end - start).count();

  // Top-k selection
  start = std::chrono::high_resolution_clock::now();
  topk(router_logits, selected_experts, top_k, n_experts, topk_scratch);
  end = std::chrono::high_resolution_clock::now();
  prof.topk_time += std::chrono::duration<double>(end - start).count();

  for (int kk = 0; kk < top_k; ++kk) {
    expert_weights[kk] = router_logits[kk];
  }

  // Process each selected expert
  for (int kk = 0; kk < top_k; ++kk) {
    int expert_id = selected_experts[kk];
    float expert_weight = expert_weights[kk];

    const float* Wg_ex = layer_weights->Wg[expert_id];
    const float* Wu_ex = layer_weights->Wu[expert_id];
    const float* Wd_ex = layer_weights->Wd[expert_id];

    // Gate projection
    start = std::chrono::high_resolution_clock::now();
    serial_matmul(x, Wg_ex, gate_out, 1, d_ff, d_model);
    end = std::chrono::high_resolution_clock::now();
    prof.gate_proj_time += std::chrono::duration<double>(end - start).count();

    // Up projection
    start = std::chrono::high_resolution_clock::now();
    serial_matmul(x, Wu_ex, up_out, 1, d_ff, d_model);
    end = std::chrono::high_resolution_clock::now();
    prof.up_proj_time += std::chrono::duration<double>(end - start).count();

    // SiLU and elementwise mul
    start = std::chrono::high_resolution_clock::now();
    silu(gate_out, d_ff);
    for (int i = 0; i < d_ff; ++i) {
      gate_out[i] *= up_out[i];
    }
    end = std::chrono::high_resolution_clock::now();
    prof.silu_mul_time += std::chrono::duration<double>(end - start).count();

    // Down projection
    start = std::chrono::high_resolution_clock::now();
    serial_matmul(gate_out, Wd_ex, expert_out, 1, d_model, d_ff);
    end = std::chrono::high_resolution_clock::now();
    prof.down_proj_time += std::chrono::duration<double>(end - start).count();

    // Add weighted to output
    for (int i = 0; i < d_model; ++i) {
      out[i] += expert_weight * expert_out[i];
    }
  }

  auto end_total = std::chrono::high_resolution_clock::now();
  prof.total_time = std::chrono::duration<double>(end_total - start_total).count();

  // Cleanup
  delete[] router_logits;
  delete[] selected_experts;
  delete[] expert_weights;
  delete[] topk_scratch;
  delete[] gate_out;
  delete[] up_out;
  delete[] expert_out;
}

// TP MoE layer using Tensor2D shards
void moe_layer_tp(float* out, const float* x,
                  Tensor2D** Wg_shards,  // [P][n_experts] each local Tensor2D (rows=d_model, cols=local_dff)
                  Tensor2D** Wu_shards,
                  Tensor2D** Wd_shards,  // [P][n_experts] (rows=local_dff, cols=d_model)
                  const float* router_w,  // Replicated [n_experts * d_model]
                  const QwenConfig* cfg, int batch_size, int P, Profiler& prof) {
  assert(batch_size == 1);
  int d_model = cfg->d_model;
  int d_ff = cfg->d_ff;
  int n_experts = cfg->n_experts;
  int top_k = cfg->top_k;
  int local_dff = d_ff / P;

  // Local buffers
  float* router_logits = new float[n_experts];
  int* selected_experts = new int[top_k];
  float* expert_weights = new float[top_k];
  float* topk_scratch = new float[n_experts];

  // Clear output
  memset(out, 0, d_model * sizeof(float));

  auto start_total = std::chrono::high_resolution_clock::now();

  // Router (replicated, serial)
  auto start = std::chrono::high_resolution_clock::now();
  serial_matmul(x, router_w, router_logits, 1, n_experts, d_model);
  auto end = std::chrono::high_resolution_clock::now();
  prof.router_time += std::chrono::duration<double>(end - start).count();

  // Top-k
  start = std::chrono::high_resolution_clock::now();
  topk(router_logits, selected_experts, top_k, n_experts, topk_scratch);
  end = std::chrono::high_resolution_clock::now();
  prof.topk_time += std::chrono::duration<double>(end - start).count();

  for (int kk = 0; kk < top_k; ++kk) {
    expert_weights[kk] = router_logits[kk];
  }

  // Per-thread local buffers
  float** local_gates = new float*[P];
  float** local_ups = new float*[P];
  float** local_acts = new float*[P];
  float** local_outs = new float*[P];
  for (int p = 0; p < P; ++p) {
    local_gates[p] = new float[local_dff];
    local_ups[p] = new float[local_dff];
    local_acts[p] = new float[local_dff];
    local_outs[p] = new float[d_model]();  // Zero-init
  }

  // Process each expert
  for (int kk = 0; kk < top_k; ++kk) {
    int expert_id = selected_experts[kk];
    float expert_weight = expert_weights[kk];

    // Launch threads
    std::vector<std::thread> threads;
    auto start_gate = std::chrono::high_resolution_clock::now();  // Time across threads, approx
    auto start_up = start_gate;
    auto start_silu = start_gate;
    auto start_down = start_gate;

    for (int p = 0; p < P; ++p) {
      threads.emplace_back([&, p, expert_id]() {
        float* lg = local_gates[p];
        float* lu = local_ups[p];
        float* la = local_acts[p];
        float* lo = local_outs[p];

        // Gate proj (column-parallel)
        serial_matmul(x, Wg_shards[p][expert_id].data, lg, 1, local_dff, d_model);

        // Up proj (column-parallel)
        serial_matmul(x, Wu_shards[p][expert_id].data, lu, 1, local_dff, d_model);

        // SiLU and mul (sharded)
        silu(lg, local_dff);
        for (int i = 0; i < local_dff; ++i) {
          la[i] = lg[i] * lu[i];
        }

        // Down proj (row-parallel, partial out)
        serial_matmul(la, Wd_shards[p][expert_id].data, lo, 1, d_model, local_dff);
      });
    }

    // Join threads
    for (auto& th : threads) th.join();

    // Accumulate timings (since parallel, time the max, but for simplicity, time the block)
    auto end_down = std::chrono::high_resolution_clock::now();
    prof.gate_proj_time += std::chrono::duration<double>(end_down - start_gate).count() / P;  // Approx per-rank
    prof.up_proj_time += std::chrono::duration<double>(end_down - start_up).count() / P;
    prof.silu_mul_time += std::chrono::duration<double>(end_down - start_silu).count() / P;
    prof.down_proj_time += std::chrono::duration<double>(end_down - start_down).count() / P;

    // Simulate comm: all-reduce (sum local_outs)
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < d_model; ++i) {
      float sum = 0.0f;
      for (int p = 0; p < P; ++p) {
        sum += local_outs[p][i];
      }
      out[i] += expert_weight * sum;
    }
    end = std::chrono::high_resolution_clock::now();
    prof.comm_time += std::chrono::duration<double>(end - start).count();

    // Reset local_outs for next expert
    for (int p = 0; p < P; ++p) {
      memset(local_outs[p], 0, d_model * sizeof(float));
    }
  }

  auto end_total = std::chrono::high_resolution_clock::now();
  prof.total_time = std::chrono::duration<double>(end_total - start_total).count();

  // Cleanup
  delete[] router_logits;
  delete[] selected_experts;
  delete[] expert_weights;
  delete[] topk_scratch;
  for (int p = 0; p < P; ++p) {
    delete[] local_gates[p];
    delete[] local_ups[p];
    delete[] local_acts[p];
    delete[] local_outs[p];
  }
  delete[] local_gates;
  delete[] local_ups;
  delete[] local_acts;
  delete[] local_outs;
}

int main(int argc, char** argv) {
  QwenConfig cfg;
  int P = 4;  // Default

  // Parse CLI --tp
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--tp" && i + 1 < argc) {
      P = std::stoi(argv[++i]);
    }
  }
  assert(cfg.d_ff % P == 0);  // For even sharding
  int local_dff = cfg.d_ff / P;
  std::cout << "TP = " << P << std::endl;
  // Random init
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  // Input x
  float* x = new float[cfg.d_model];
  for (int i = 0; i < cfg.d_model; ++i) x[i] = dist(gen);

  // Full weights
  float* full_router = new float[cfg.n_experts * cfg.d_model];
  for (int i = 0; i < cfg.n_experts * cfg.d_model; ++i) full_router[i] = dist(gen);

  float** full_Wg = new float*[cfg.n_experts];
  float** full_Wu = new float*[cfg.n_experts];
  float** full_Wd = new float*[cfg.n_experts];
  for (int e = 0; e < cfg.n_experts; ++e) {
    full_Wg[e] = new float[cfg.d_ff * cfg.d_model];
    full_Wu[e] = new float[cfg.d_ff * cfg.d_model];
    full_Wd[e] = new float[cfg.d_model * cfg.d_ff];
    for (int i = 0; i < cfg.d_ff * cfg.d_model; ++i) {
      full_Wg[e][i] = dist(gen);
      full_Wu[e][i] = dist(gen);
    }
    for (int i = 0; i < cfg.d_model * cfg.d_ff; ++i) {
      full_Wd[e][i] = dist(gen);
    }
  }

  // Serial weights
  QwenLayerWeights weights_serial;
  weights_serial.router_w = full_router;
  weights_serial.Wg = full_Wg;
  weights_serial.Wu = full_Wu;
  weights_serial.Wd = full_Wd;

  // Shard for TP using Tensor2D
  Tensor2D** Wg_shards = new Tensor2D*[P];
  Tensor2D** Wu_shards = new Tensor2D*[P];
  Tensor2D** Wd_shards = new Tensor2D*[P];
  for (int p = 0; p < P; ++p) {
    Wg_shards[p] = new Tensor2D[cfg.n_experts];
    Wu_shards[p] = new Tensor2D[cfg.n_experts];
    Wd_shards[p] = new Tensor2D[cfg.n_experts];
    for (int e = 0; e < cfg.n_experts; ++e) {
      // Allocate local data
      float* local_wg = new float[cfg.d_model * local_dff];
      float* local_wu = new float[cfg.d_model * local_dff];
      float* local_wd = new float[local_dff * cfg.d_model];

      // Shard Wg/Wu (column-shard: extract columns p*local_dff to (p+1)*local_dff)
      for (int r = 0; r < cfg.d_model; ++r) {
        for (int lc = 0; lc < local_dff; ++lc) {
          int gc = p * local_dff + lc;
          local_wg[r * local_dff + lc] = full_Wg[e][r * cfg.d_ff + gc];
          local_wu[r * local_dff + lc] = full_Wu[e][r * cfg.d_ff + gc];
        }
      }

      // Shard Wd (row-shard: extract rows p*local_dff to (p+1)*local_dff)
      for (int lr = 0; lr < local_dff; ++lr) {
        for (int c = 0; c < cfg.d_model; ++c) {
          int gr = p * local_dff + lr;
          local_wd[lr * cfg.d_model + c] = full_Wd[e][gr * cfg.d_model + c];
        }
      }

      Wg_shards[p][e] = Tensor2D(local_wg, cfg.d_model, local_dff);  // Note: rows=d_model, cols=local_dff (but matmul uses transposed view implicitly)
      Wu_shards[p][e] = Tensor2D(local_wu, cfg.d_model, local_dff);
      Wd_shards[p][e] = Tensor2D(local_wd, local_dff, cfg.d_model);
    }
  }

  // If P==1, shards[0][e] would be full, but since P=1 local_dff=d_ff, it's the same.

  // Outputs
  float* out_serial = new float[cfg.d_model];
  float* out_tp = new float[cfg.d_model];

  // Serial run
  Profiler prof_serial;
  moe_layer_serial(out_serial, x, &weights_serial, &cfg, 1, prof_serial);
  prof_serial.print("Serial");

  // TP run
  Profiler prof_tp;
  moe_layer_tp(out_tp, x, Wg_shards, Wu_shards, Wd_shards, full_router, &cfg, 1, P, prof_tp);
  prof_tp.print("TP");

  // Speedup
  double speedup = prof_serial.total_time / prof_tp.total_time;
  std::cout << "Speedup (Serial / TP): " << speedup << "x" << std::endl;

  // Validate
  float tol = 1e-4f;
  bool match = true;
  for (int i = 0; i < cfg.d_model; ++i) {
    if (std::abs(out_serial[i] - out_tp[i]) > tol) {
      match = false;
      break;
    }
  }
  std::cout << "Results match: " << (match ? "Yes" : "No") << std::endl;

  // Cleanup
  delete[] x;
  delete[] full_router;
  for (int e = 0; e < cfg.n_experts; ++e) {
    delete[] full_Wg[e];
    delete[] full_Wu[e];
    delete[] full_Wd[e];
  }
  delete[] full_Wg;
  delete[] full_Wu;
  delete[] full_Wd;

  for (int p = 0; p < P; ++p) {
    for (int e = 0; e < cfg.n_experts; ++e) {
      delete[] Wg_shards[p][e].data;
      delete[] Wu_shards[p][e].data;
      delete[] Wd_shards[p][e].data;
    }
    delete[] Wg_shards[p];
    delete[] Wu_shards[p];
    delete[] Wd_shards[p];
  }
  delete[] Wg_shards;
  delete[] Wu_shards;
  delete[] Wd_shards;

  delete[] out_serial;
  delete[] out_tp;

  return 0;
}
