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
#include <mpi.h>
#include <sys/stat.h>
#include <filesystem>

// Hardcoded config
struct QwenConfig {
  int d_model = 2048;
  int d_ff = 768;
  int n_experts = 128;
  int top_k = 8;
};

// Layer weights struct (simplified, no biases)
struct QwenLayerWeights {
  float* router_w = nullptr;  // [n_experts * d_model] (row-major, rows=n_experts, cols=d_model)
  float** Wg = nullptr;       // Array of [d_model * d_ff] (rows=d_model, cols=d_ff) per expert
  float** Wu = nullptr;
  float** Wd = nullptr;       // [d_ff * d_model] (rows=d_ff, cols=d_model)
};

// Tensor shard struct: represents a 2D tensor (row-major flat array)
struct Tensor2D {
  float* data;  // Flat data pointer
  int rows;     // Number of rows
  int cols;     // Number of columns

  // Constructor
  Tensor2D(float* d = nullptr, int r = 0, int c = 0) : data(d), rows(r), cols(c) {}

  // Size in elements
  size_t size() const { return static_cast<size_t>(rows) * cols; }

  // Access element (i,j)
  float& at(int i, int j) { return data[i * cols + j]; }
  const float& at(int i, int j) const { return data[i * cols + j]; }
};

// Profiling struct to hold timings
struct Profiler {
  double router_time = 0.0;
  double topk_time = 0.0;
  double gate_proj_time = 0.0;
  double up_proj_time = 0.0;
  double silu_mul_time = 0.0;
  double down_proj_time = 0.0;
  double comm_time = 0.0;  // Only for TP (reduce)
  double total_time = 0.0;

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

// Helper functions for weights directory management
void ensure_weights_dir() {
  std::filesystem::create_directory("weights");
}

bool weights_exist(const QwenConfig* cfg) {
  int n_experts = cfg->n_experts;
  
  // Check if router file exists
  if (!std::filesystem::exists("weights/router.bin")) {
    return false;
  }
  
  // Check if all expert files exist
  for (int e = 0; e < n_experts; ++e) {
    std::string base = "weights/expert" + std::to_string(e) + "_";
    if (!std::filesystem::exists(base + "Wg.bin") ||
        !std::filesystem::exists(base + "Wu.bin") ||
        !std::filesystem::exists(base + "Wd.bin")) {
      return false;
    }
  }
  
  return true;
}

// Serial matmul: C = A * B (m x k) * (k x n) -> (m x n)
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

// SiLU activation in-place
void silu(float* x, int n) {
  for (int i = 0; i < n; ++i) {
    x[i] = x[i] / (1.0f + expf(-x[i]));
  }
}

// Top-k selection with softmax normalization (modifies logits in-place for top-k, uses scratch for copy)
void topk(float* logits, int* indices, int k, int n, float* scratch) {
  // Copy logits to scratch
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
  int d_model = cfg->d_model;
  int d_ff = cfg->d_ff;
  int n_experts = cfg->n_experts;
  int top_k = cfg->top_k;

  // Local buffers (heap-allocated) - sized for batch
  float* router_logits = new float[batch_size * n_experts];
  int* selected_experts = new int[batch_size * top_k];
  float* expert_weights = new float[batch_size * top_k];
  float* topk_scratch = new float[n_experts];  // Reused per token
  float* gate_out = new float[d_ff];  // Reused per expert per token
  float* up_out = new float[d_ff];    // Reused per expert per token
  float* expert_out = new float[d_model];  // Reused per expert per token

  // Clear output
  memset(out, 0, batch_size * d_model * sizeof(float));

  auto start_total = std::chrono::high_resolution_clock::now();

  // Process each token in the batch
  for (int t = 0; t < batch_size; ++t) {
    const float* token_x = x + t * d_model;
    float* token_out = out + t * d_model;
    float* token_router_logits = router_logits + t * n_experts;
    int* token_selected_experts = selected_experts + t * top_k;
    float* token_expert_weights = expert_weights + t * top_k;
    
    // Router computation for this token
    auto start = std::chrono::high_resolution_clock::now();
    serial_matmul(token_x, layer_weights->router_w, token_router_logits, 1, n_experts, d_model);
    auto end = std::chrono::high_resolution_clock::now();
    prof.router_time += std::chrono::duration<double>(end - start).count();

    // Top-k selection for this token
    start = std::chrono::high_resolution_clock::now();
    topk(token_router_logits, token_selected_experts, top_k, n_experts, topk_scratch);
    end = std::chrono::high_resolution_clock::now();
    prof.topk_time += std::chrono::duration<double>(end - start).count();

    // Copy selected weights
    for (int kk = 0; kk < top_k; ++kk) {
      token_expert_weights[kk] = token_router_logits[token_selected_experts[kk]];
    }

    // Process each selected expert for this token
    for (int kk = 0; kk < top_k; ++kk) {
      int expert_id = token_selected_experts[kk];
      float expert_weight = token_expert_weights[kk];

      const float* Wg_ex = layer_weights->Wg[expert_id];
      const float* Wu_ex = layer_weights->Wu[expert_id];
      const float* Wd_ex = layer_weights->Wd[expert_id];

      // Gate projection
      auto start = std::chrono::high_resolution_clock::now();
      serial_matmul(token_x, Wg_ex, gate_out, 1, d_ff, d_model);
      auto end = std::chrono::high_resolution_clock::now();
      prof.gate_proj_time += std::chrono::duration<double>(end - start).count();

      // Up projection
      start = std::chrono::high_resolution_clock::now();
      serial_matmul(token_x, Wu_ex, up_out, 1, d_ff, d_model);
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
      prof.down_proj_time += std::chrono::duration<double>(end - start).count();

      // Add weighted to output
      for (int i = 0; i < d_model; ++i) {
        token_out[i] += expert_weight * expert_out[i];
      }
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
                  Tensor2D* Wg_shards,  // flat [local_p * n_experts], rows=d_model, cols=local_dff
                  Tensor2D* Wu_shards,
                  Tensor2D* Wd_shards,  // rows=local_dff, cols=d_model
                  const float* router_w,  // Replicated
                  const QwenConfig* cfg, int batch_size, int local_p, int mpi_rank, int mpi_size, Profiler& prof) {
  int d_model = cfg->d_model;
  int d_ff = cfg->d_ff;
  int n_experts = cfg->n_experts;
  int top_k = cfg->top_k;
  int local_dff = d_ff / (local_p * mpi_size);  // Global TP size = local_p * mpi_size

  // Local buffers - sized for batch
  float* router_logits = new float[batch_size * n_experts];
  int* selected_experts = new int[batch_size * top_k];
  float* expert_weights = new float[batch_size * top_k];
  float* topk_scratch = new float[n_experts];  // Reused per token

  // Clear output
  memset(out, 0, batch_size * d_model * sizeof(float));

  auto start_total = std::chrono::high_resolution_clock::now();

  // Process each token in the batch
  for (int t = 0; t < batch_size; ++t) {
    const float* token_x = x + t * d_model;
    float* token_out = out + t * d_model;
    float* token_router_logits = router_logits + t * n_experts;
    int* token_selected_experts = selected_experts + t * top_k;
    float* token_expert_weights = expert_weights + t * top_k;

    // Router (replicated, serial) for this token
    auto start = std::chrono::high_resolution_clock::now();
    serial_matmul(token_x, router_w, token_router_logits, 1, n_experts, d_model);
    auto end = std::chrono::high_resolution_clock::now();
    prof.router_time += std::chrono::duration<double>(end - start).count();

    // Top-k for this token
    start = std::chrono::high_resolution_clock::now();
    topk(token_router_logits, token_selected_experts, top_k, n_experts, topk_scratch);
    end = std::chrono::high_resolution_clock::now();
    prof.topk_time += std::chrono::duration<double>(end - start).count();

    // Copy selected weights
    for (int kk = 0; kk < top_k; ++kk) {
      token_expert_weights[kk] = token_router_logits[token_selected_experts[kk]];
    }

    // Per-thread local buffers
    float** local_gates = new float*[local_p];
    float** local_ups = new float*[local_p];
    float** local_acts = new float*[local_p];
    float** local_outs = new float*[local_p];
    for (int lp = 0; lp < local_p; ++lp) {
      local_gates[lp] = new float[local_dff];
      local_ups[lp] = new float[local_dff];
      local_acts[lp] = new float[local_dff];
      local_outs[lp] = new float[d_model]();  // Zero-init
    }

    // Process each expert for this token
    for (int kk = 0; kk < top_k; ++kk) {
      int expert_id = token_selected_experts[kk];
      float expert_weight = token_expert_weights[kk];

      // Launch threads
      std::vector<std::thread> threads;
      auto start_gate = std::chrono::high_resolution_clock::now();

      for (int lp = 0; lp < local_p; ++lp) {
        threads.emplace_back([&, lp, expert_id]() {
          float* lg = local_gates[lp];
          float* lu = local_ups[lp];
          float* la = local_acts[lp];
          float* lo = local_outs[lp];

          // Gate proj (column-parallel)
          serial_matmul(token_x, Wg_shards[lp * n_experts + expert_id].data, lg, 1, local_dff, d_model);

          // Up proj (column-parallel)
          serial_matmul(token_x, Wu_shards[lp * n_experts + expert_id].data, lu, 1, local_dff, d_model);

          // SiLU and mul (sharded)
          silu(lg, local_dff);
          for (int i = 0; i < local_dff; ++i) {
            la[i] = lg[i] * lu[i];
          }

          // Down proj (row-parallel, partial out)
          serial_matmul(la, Wd_shards[lp * n_experts + expert_id].data, lo, 1, d_model, local_dff);
        });
      }

      // Join threads
      for (auto& th : threads) th.join();

      // Accumulate timings (approx)
      auto end_down = std::chrono::high_resolution_clock::now();
      double block_time = std::chrono::duration<double>(end_down - start_gate).count() / local_p;
      prof.gate_proj_time += block_time;
      prof.up_proj_time += block_time;
      prof.silu_mul_time += block_time;
      prof.down_proj_time += block_time;

      // Local reduce (sum within rank)
      float* rank_out = new float[d_model]();
      for (int i = 0; i < d_model; ++i) {
        for (int lp = 0; lp < local_p; ++lp) {
          rank_out[i] += local_outs[lp][i];
        }
      }

      // Global comm: MPI_Allreduce sum across ranks
      auto start_comm = std::chrono::high_resolution_clock::now();
      float* global_sum = new float[d_model];
      MPI_Allreduce(rank_out, global_sum, d_model, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      auto end_comm = std::chrono::high_resolution_clock::now();
      prof.comm_time += std::chrono::duration<double>(end_comm - start_comm).count();

      // Add weighted to token output
      for (int i = 0; i < d_model; ++i) {
        token_out[i] += expert_weight * global_sum[i];
      }

      delete[] rank_out;
      delete[] global_sum;

      // Reset local_outs for next expert
      for (int lp = 0; lp < local_p; ++lp) {
        memset(local_outs[lp], 0, d_model * sizeof(float));
      }
    }

    // Cleanup local buffers for this token
    for (int lp = 0; lp < local_p; ++lp) {
      delete[] local_gates[lp];
      delete[] local_ups[lp];
      delete[] local_acts[lp];
      delete[] local_outs[lp];
    }
    delete[] local_gates;
    delete[] local_ups;
    delete[] local_acts;
    delete[] local_outs;
  } // End batch loop

  auto end_total = std::chrono::high_resolution_clock::now();
  prof.total_time = std::chrono::duration<double>(end_total - start_total).count();

  // Cleanup
  delete[] router_logits;
  delete[] selected_experts;
  delete[] expert_weights;
  delete[] topk_scratch;
}

// Save full weights to disk (binary files per expert per weight)
void save_full_weights(const QwenConfig* cfg, float* full_router, float** full_Wg, float** full_Wu, float** full_Wd) {
  int n_experts = cfg->n_experts;
  int d_model = cfg->d_model;
  int d_ff = cfg->d_ff;

  ensure_weights_dir();

  // Save router
  std::ofstream out_router("weights/router.bin", std::ios::binary);
  out_router.write((char*)full_router, n_experts * d_model * sizeof(float));
  out_router.close();

  for (int e = 0; e < n_experts; ++e) {
    std::string base = "weights/expert" + std::to_string(e) + "_";
    std::ofstream out_wg(base + "Wg.bin", std::ios::binary);
    out_wg.write((char*)full_Wg[e], d_model * d_ff * sizeof(float));
    out_wg.close();

    std::ofstream out_wu(base + "Wu.bin", std::ios::binary);
    out_wu.write((char*)full_Wu[e], d_model * d_ff * sizeof(float));
    out_wu.close();

    std::ofstream out_wd(base + "Wd.bin", std::ios::binary);
    out_wd.write((char*)full_Wd[e], d_ff * d_model * sizeof(float));
    out_wd.close();
  }
}

// Load full weights from disk (binary files per expert per weight)
void load_full_weights(const QwenConfig* cfg, float* full_router, float** full_Wg, float** full_Wu, float** full_Wd) {
  int n_experts = cfg->n_experts;
  int d_model = cfg->d_model;
  int d_ff = cfg->d_ff;

  // Load router
  std::ifstream in_router("weights/router.bin", std::ios::binary);
  in_router.read((char*)full_router, n_experts * d_model * sizeof(float));
  in_router.close();

  for (int e = 0; e < n_experts; ++e) {
    std::string base = "weights/expert" + std::to_string(e) + "_";
    
    std::ifstream in_wg(base + "Wg.bin", std::ios::binary);
    in_wg.read((char*)full_Wg[e], d_model * d_ff * sizeof(float));
    in_wg.close();

    std::ifstream in_wu(base + "Wu.bin", std::ios::binary);
    in_wu.read((char*)full_Wu[e], d_model * d_ff * sizeof(float));
    in_wu.close();

    std::ifstream in_wd(base + "Wd.bin", std::ios::binary);
    in_wd.read((char*)full_Wd[e], d_ff * d_model * sizeof(float));
    in_wd.close();
  }
}

// Load local shards for this MPI rank's threads
void load_local_shards(const QwenConfig* cfg, int local_p, int mpi_rank, int mpi_size, Tensor2D* Wg_shards, Tensor2D* Wu_shards, Tensor2D* Wd_shards) {
  int n_experts = cfg->n_experts;
  int d_model = cfg->d_model;
  int d_ff = cfg->d_ff;
  int total_p = local_p * mpi_size;
  int local_dff = d_ff / total_p;

  std::vector<std::thread> load_threads;
  for (int lp = 0; lp < local_p; ++lp) {
    load_threads.emplace_back([&, lp]() {
      int global_p = mpi_rank * local_p + lp;
      for (int e = 0; e < n_experts; ++e) {
        std::string base = "weights/expert" + std::to_string(e) + "_";

        // Alloc
        float* local_wg = new float[d_model * local_dff];
        float* local_wu = new float[d_model * local_dff];
        float* local_wd = new float[local_dff * d_model];

        // Load Wg shard (column, non-contig)
        std::ifstream in_wg(base + "Wg.bin", std::ios::binary);
        for (int r = 0; r < d_model; ++r) {
          in_wg.seekg((r * d_ff + global_p * local_dff) * sizeof(float));
          in_wg.read((char*)(local_wg + r * local_dff), local_dff * sizeof(float));
        }

        // Load Wu shard (similar)
        std::ifstream in_wu(base + "Wu.bin", std::ios::binary);
        for (int r = 0; r < d_model; ++r) {
          in_wu.seekg((r * d_ff + global_p * local_dff) * sizeof(float));
          in_wu.read((char*)(local_wu + r * local_dff), local_dff * sizeof(float));
        }

        // Load Wd shard (row, contig)
        std::ifstream in_wd(base + "Wd.bin", std::ios::binary);
        in_wd.seekg((global_p * local_dff * d_model) * sizeof(float));
        in_wd.read((char*)local_wd, local_dff * d_model * sizeof(float));

        // Assign
        int idx = lp * n_experts + e;
        Wg_shards[idx] = Tensor2D(local_wg, d_model, local_dff);
        Wu_shards[idx] = Tensor2D(local_wu, d_model, local_dff);
        Wd_shards[idx] = Tensor2D(local_wd, local_dff, d_model);
      }
    });
  }
  for (auto& th : load_threads) th.join();
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  Profiler prof_serial;

  QwenConfig cfg;
  int total_tp = 4;  // Default
  int batch_size = 1;  // Default

  // Parse CLI --tp (total TP degree) and --batch
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--tp" && i + 1 < argc) {
      total_tp = std::stoi(argv[++i]);
    } else if (std::string(argv[i]) == "--batch" && i + 1 < argc) {
      batch_size = std::stoi(argv[++i]);
    }
  }
  assert(cfg.d_ff % total_tp == 0);  // For even sharding
  assert(total_tp % mpi_size == 0);
  int local_p = total_tp / mpi_size;

  float* out_serial = new float[batch_size * cfg.d_model];
  if (mpi_rank == 0) {
    // Random init and save full weights (only master generates and saves)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

    // Input x (broadcast later)
    float* x = new float[batch_size * cfg.d_model];
    for (int i = 0; i < batch_size * cfg.d_model; ++i) x[i] = dist(gen);

    // Allocate full weights
    float* full_router = new float[cfg.n_experts * cfg.d_model];
    float** full_Wg = new float*[cfg.n_experts];
    float** full_Wu = new float*[cfg.n_experts];
    float** full_Wd = new float*[cfg.n_experts];
    for (int e = 0; e < cfg.n_experts; ++e) {
      full_Wg[e] = new float[cfg.d_model * cfg.d_ff];
      full_Wu[e] = new float[cfg.d_model * cfg.d_ff];
      full_Wd[e] = new float[cfg.d_ff * cfg.d_model];
    }

    // Check if weights exist, if not generate and save them
    if (weights_exist(&cfg)) {
      std::cout << "Loading existing weights from weights/ directory..." << std::endl;
      load_full_weights(&cfg, full_router, full_Wg, full_Wu, full_Wd);
    } else {
      std::cout << "Generating new weights and saving to weights/ directory..." << std::endl;
      for (int i = 0; i < cfg.n_experts * cfg.d_model; ++i) full_router[i] = dist(gen);
      
      for (int e = 0; e < cfg.n_experts; ++e) {
        for (int i = 0; i < cfg.d_model * cfg.d_ff; ++i) {
          full_Wg[e][i] = dist(gen);
          full_Wu[e][i] = dist(gen);
        }
        for (int i = 0; i < cfg.d_ff * cfg.d_model; ++i) {
          full_Wd[e][i] = dist(gen);
        }
      }
      
      // Save to disk
      save_full_weights(&cfg, full_router, full_Wg, full_Wu, full_Wd);
    }

    // Serial run on master (with full)
    QwenLayerWeights weights_serial;
    weights_serial.router_w = full_router;
    weights_serial.Wg = full_Wg;
    weights_serial.Wu = full_Wu;
    weights_serial.Wd = full_Wd;

    moe_layer_serial(out_serial, x, &weights_serial, &cfg, batch_size, prof_serial);
    prof_serial.print("Serial");

    // Cleanup full (master also loads its shards later)
    for (int e = 0; e < cfg.n_experts; ++e) {
      delete[] full_Wg[e];
      delete[] full_Wu[e];
      delete[] full_Wd[e];
    }
    delete[] full_Wg;
    delete[] full_Wu;
    delete[] full_Wd;
  }

  // Barrier to ensure files written
  MPI_Barrier(MPI_COMM_WORLD);

  // All ranks load router (small)
  float* router_w = new float[cfg.n_experts * cfg.d_model];
  std::ifstream in_router("weights/router.bin", std::ios::binary);
  in_router.read((char*)router_w, cfg.n_experts * cfg.d_model * sizeof(float));
  in_router.close();

  // Load local shards
  Tensor2D* Wg_shards = new Tensor2D[local_p * cfg.n_experts];
  Tensor2D* Wu_shards = new Tensor2D[local_p * cfg.n_experts];
  Tensor2D* Wd_shards = new Tensor2D[local_p * cfg.n_experts];
  load_local_shards(&cfg, local_p, mpi_rank, mpi_size, Wg_shards, Wu_shards, Wd_shards);

  // Input x (broadcast from master)
  float* x = new float[batch_size * cfg.d_model];
  MPI_Bcast(x, batch_size * cfg.d_model, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // TP run
  float* out_tp = new float[batch_size * cfg.d_model];
  Profiler prof_tp;
  moe_layer_tp(out_tp, x, Wg_shards, Wu_shards, Wd_shards, router_w, &cfg, batch_size, local_p, mpi_rank, mpi_size, prof_tp);
  if (mpi_rank == 0) {
    prof_tp.print("TP");

    // Speedup (serial on master)
    double speedup = prof_serial.total_time / prof_tp.total_time;
    std::cout << "Speedup (Serial / TP): " << speedup << "x" << std::endl;
    float tol = 1e-4f;
    bool match = true;
    for (int i = 0; i < batch_size * cfg.d_model; i++) {
      if (std::abs(out_serial[i] - out_tp[i]) > tol) {
        match = false;
        break;
      }
    }
    std::cout << "Results match: " << (match ? "Yes" : "No") << std::endl;

  }

  delete[] out_serial;
  // Cleanup
  delete[] x;
  delete[] router_w;
  delete[] out_tp;
  for (int i = 0; i < local_p * cfg.n_experts; ++i) {
    delete[] Wg_shards[i].data;
    delete[] Wu_shards[i].data;
    delete[] Wd_shards[i].data;
  }
  delete[] Wg_shards;
  delete[] Wu_shards;
  delete[] Wd_shards;

  MPI_Finalize();
  return 0;
}
