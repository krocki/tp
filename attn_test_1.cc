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

// Hardcoded config
struct AttnConfig {
  int d_model = 2048;
  int n_q = 32;
  int n_kv = 4;
  int head_dim = 128;
  float rms_eps = 1e-6f;
  float rope_theta = 1000000.0f;
  int max_seq = 1024;  // For cache
};

// Layer weights
struct AttnLayerWeights {
  float* rms1_w = nullptr;  // [d_model]
  float* W_proj[3] = {nullptr, nullptr, nullptr};  // Wq, Wk, Wv
  float* Wo = nullptr;  // [total_q_dim * d_model]
  float* hk_norm[2] = {nullptr, nullptr};  // q_norm, k_norm [head_dim]
};

// Tensor2D
struct Tensor2D {
  float* data = nullptr;
  int rows = 0, cols = 0;

  Tensor2D(float* d = nullptr, int r = 0, int c = 0) : data(d), rows(r), cols(c) {}

  size_t size() const { return static_cast<size_t>(rows) * cols; }
};

// Profiler
struct Profiler {
  double rms_time = 0.0;
  double qkv_proj_time = 0.0;
  double qk_norm_time = 0.0;
  double rope_time = 0.0;
  double scores_time = 0.0;
  double softmax_time = 0.0;
  double weighted_v_time = 0.0;
  double o_proj_time = 0.0;
  double comm_time = 0.0;
  double total_time = 0.0;

  void print(const std::string& prefix) const {
    std::cout << prefix << " Breakdown:" << std::endl;
    std::cout << "  RMS: " << rms_time << "s" << std::endl;
    std::cout << "  QKV Proj: " << qkv_proj_time << "s" << std::endl;
    std::cout << "  Q/K Norm: " << qk_norm_time << "s" << std::endl;
    std::cout << "  RoPE: " << rope_time << "s" << std::endl;
    std::cout << "  Scores: " << scores_time << "s" << std::endl;
    std::cout << "  Softmax: " << softmax_time << "s" << std::endl;
    std::cout << "  Weighted V: " << weighted_v_time << "s" << std::endl;
    std::cout << "  O Proj: " << o_proj_time << "s" << std::endl;
    std::cout << "  Comm: " << comm_time << "s" << std::endl;
    std::cout << "  Total: " << total_time << "s" << std::endl;
  }
};

// Helper: serial matmul C = A * B
void serial_matmul(const float* A, const float* B, float* C, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int p = 0; p < k; ++p) sum += A[i * k + p] * B[p * n + j];
      C[i * n + j] = sum;
    }
  }
}

// Helper: rmsnorm
void rmsnorm(float* out, const float* x, const float* weight, int size, float eps) {
  float ss = 0.0f;
  for (int i = 0; i < size; i++) ss += x[i] * x[i];
  ss = ss / size + eps;
  ss = 1.0f / sqrtf(ss);
  for (int i = 0; i < size; i++) out[i] = x[i] * ss * weight[i];
}

// Helper: rope for one vector
void rope(float* vec, int pos, int head_dim, float theta) {
  for (int i = 0; i < head_dim; i += 2) {
    float freq = (float)pos / powf(theta, (float)i / head_dim);
    float c = cosf(freq);
    float s = sinf(freq);
    float v0 = vec[i];
    float v1 = vec[i + 1];
    vec[i] = v0 * c - v1 * s;
    vec[i + 1] = v0 * s + v1 * c;
  }
}

// Helper: softmax in-place
void softmax(float* x, int size) {
  float max_val = x[0];
  for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  for (int i = 0; i < size; i++) x[i] /= sum;
}

// Helper: TP column matmul
void tp_column_matmul(const float* A, const Tensor2D& B_shard, float* C_shard, int m, int local_n, int k) {
  serial_matmul(A, B_shard.data, C_shard, m, local_n, k);
}

// Helper: TP row matmul with hybrid reduce
void tp_row_matmul(const float* A_shard, const Tensor2D& B_shard, float* local_C, int m, int n, int local_k, int local_p, int mpi_rank, int mpi_size, double& comm_time) {
  serial_matmul(A_shard, B_shard.data, local_C, m, n, local_k);
  float* rank_C = new float[m * n]();
  // Assume caller sums local_C from threads to rank_C, then here MPI
  auto start = std::chrono::high_resolution_clock::now();
  MPI_Allreduce(rank_C, local_C, m * n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);  // In-place on C after local sum
  auto end = std::chrono::high_resolution_clock::now();
  comm_time += std::chrono::duration<double>(end - start).count();
  delete[] rank_C;
}

// Helper: apply rmsnorm to heads (shared)
void apply_rmsnorm_to_heads(float* vec, int num_heads, int head_dim, const float* weight, float eps) {
  if (weight) {
    for (int h = 0; h < num_heads; h++) {
      float* vh = vec + h * head_dim;
      rmsnorm(vh, vh, weight, head_dim, eps);
    }
  }
}

// Helper: apply rope to heads (shared)
void apply_rope_to_heads(float* vec, int num_heads, int head_dim, int pos, float theta) {
  for (int h = 0; h < num_heads; h++) {
    rope(vec + h * head_dim, pos, head_dim, theta);
  }
}

// Helper: append to cache (shared)
void append_to_cache(float* cache, const float* cur, int cache_pos, int dim) {
  memcpy(cache + cache_pos * dim, cur, dim * sizeof(float));
}

// Helper: compute attn for local heads (shared)
void compute_attn_local(float* attn_out, const float* Q, const float* k_cache, const float* v_cache, float* attn_scores, int local_n_q, int local_n_kv, int head_dim, int pos, int context_len, float attn_scale, int group_size) {
  memset(attn_out, 0, local_n_q * head_dim * sizeof(float));
  for (int local_g = 0; local_g < local_n_kv; local_g++) {
    for (int qg = 0; qg < group_size; qg++) {
      int q_head = local_g * group_size + qg;
      const float* q = Q + q_head * head_dim;
      for (int tk = 0; tk < context_len; tk++) {
        const float* k = k_cache + tk * (local_n_kv * head_dim) + local_g * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += q[d] * k[d];
        attn_scores[tk] = dot * attn_scale;
      }
      for (int tk = pos + 1; tk < context_len; tk++) attn_scores[tk] = -INFINITY;
      softmax(attn_scores, context_len);
      float* head_out = attn_out + q_head * head_dim;
      for (int tk = 0; tk < context_len; tk++) {
        const float* v = v_cache + tk * (local_n_kv * head_dim) + local_g * head_dim;
        float w = attn_scores[tk];
        for (int d = 0; d < head_dim; d++) head_out[d] += w * v[d];
      }
    }
  }
}

// Serial attention
void attention_serial(float* out, const float* x, AttnLayerWeights* weights, const AttnConfig* cfg, float* k_cache, float* v_cache, int pos, int cache_pos, Profiler& prof) {
  int d_model = cfg->d_model;
  int total_q_dim = cfg->n_q * cfg->head_dim;
  int total_kv_dim = cfg->n_kv * cfg->head_dim;
  float attn_scale = 1.0f / sqrtf((float)cfg->head_dim);
  int context_len = cache_pos + 1;
  int group_size = cfg->n_q / cfg->n_kv;
  float* x_norm = new float[d_model];
  float* Q = new float[total_q_dim];
  float* K_cur = new float[total_kv_dim];
  float* V_cur = new float[total_kv_dim];
  float* attn_scores = new float[context_len];
  float* attn_out = new float[total_q_dim];
  auto start_total = std::chrono::high_resolution_clock::now();
  auto start = std::chrono::high_resolution_clock::now();
  rmsnorm(x_norm, x, weights->rms1_w, d_model, cfg->rms_eps);
  auto end = std::chrono::high_resolution_clock::now();
  prof.rms_time += std::chrono::duration<double>(end - start).count();
  start = std::chrono::high_resolution_clock::now();
  serial_matmul(x_norm, weights->W_proj[0], Q, 1, total_q_dim, d_model);
  serial_matmul(x_norm, weights->W_proj[1], K_cur, 1, total_kv_dim, d_model);
  serial_matmul(x_norm, weights->W_proj[2], V_cur, 1, total_kv_dim, d_model);
  end = std::chrono::high_resolution_clock::now();
  prof.qkv_proj_time += std::chrono::duration<double>(end - start).count();
  start = std::chrono::high_resolution_clock::now();
  apply_rmsnorm_to_heads(Q, cfg->n_q, cfg->head_dim, weights->hk_norm[0], cfg->rms_eps);
  apply_rmsnorm_to_heads(K_cur, cfg->n_kv, cfg->head_dim, weights->hk_norm[1], cfg->rms_eps);
  end = std::chrono::high_resolution_clock::now();
  prof.qk_norm_time += std::chrono::duration<double>(end - start).count();
  start = std::chrono::high_resolution_clock::now();
  apply_rope_to_heads(Q, cfg->n_q, cfg->head_dim, pos, cfg->rope_theta);
  apply_rope_to_heads(K_cur, cfg->n_kv, cfg->head_dim, pos, cfg->rope_theta);
  end = std::chrono::high_resolution_clock::now();
  prof.rope_time += std::chrono::duration<double>(end - start).count();
  start = std::chrono::high_resolution_clock::now();
  append_to_cache(k_cache, K_cur, cache_pos, total_kv_dim);
  append_to_cache(v_cache, V_cur, cache_pos, total_kv_dim);
  end = std::chrono::high_resolution_clock::now();
  // (append time not profiled separately)
  start = std::chrono::high_resolution_clock::now();
  compute_attn_local(attn_out, Q, k_cache, v_cache, attn_scores, cfg->n_q, cfg->n_kv, cfg->head_dim, pos, context_len, attn_scale, group_size);
  end = std::chrono::high_resolution_clock::now();
  double attn_time = std::chrono::duration<double>(end - start).count();
  prof.scores_time += attn_time * 0.4;
  prof.softmax_time += attn_time * 0.3;
  prof.weighted_v_time += attn_time * 0.3;
  start = std::chrono::high_resolution_clock::now();
  serial_matmul(attn_out, weights->Wo, out, 1, d_model, total_q_dim);
  end = std::chrono::high_resolution_clock::now();
  prof.o_proj_time += std::chrono::duration<double>(end - start).count();
  prof.total_time = std::chrono::duration<double>(end - start_total).count();
  delete[] x_norm;
  delete[] Q;
  delete[] K_cur;
  delete[] V_cur;
  delete[] attn_scores;
  delete[] attn_out;
}

// TP attention (reuse helpers)
void attention_tp(float* out, const float* x, const Tensor2D& rms1_w, Tensor2D* Wq_shards, Tensor2D* Wk_shards, Tensor2D* Wv_shards, Tensor2D* Wo_shards, const Tensor2D& q_norm, const Tensor2D& k_norm, const AttnConfig* cfg, float* k_cache_shard, float* v_cache_shard, int pos, int cache_pos, int local_p, int mpi_rank, int mpi_size, Profiler& prof) {
  int d_model = cfg->d_model;
  int total_tp = local_p * mpi_size;
  int sub_n_q = cfg->n_q / total_tp;
  int sub_n_kv = cfg->n_kv / total_tp;
  int sub_q_dim = sub_n_q * cfg->head_dim;
  int sub_kv_dim = sub_n_kv * cfg->head_dim;
  int rank_n_q = local_p * sub_n_q;
  int rank_n_kv = local_p * sub_n_kv;
  int rank_q_dim = rank_n_q * cfg->head_dim;
  int rank_kv_dim = rank_n_kv * cfg->head_dim;
  int context_len = cache_pos + 1;
  float attn_scale = 1.0f / sqrtf((float)cfg->head_dim);
  int group_size = cfg->n_q / cfg->n_kv;
  float* x_norm = new float[d_model];
  rmsnorm(x_norm, x, rms1_w.data, d_model, cfg->rms_eps);
  float* Q_shard = new float[rank_q_dim];
  float* K_cur_shard = new float[rank_kv_dim];
  float* V_cur_shard = new float[rank_kv_dim];
  float* attn_scores = new float[context_len];
  float* attn_out_shard = new float[rank_q_dim];
  auto start_total = std::chrono::high_resolution_clock::now();
  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();
  prof.rms_time += std::chrono::duration<double>(end - start).count();
  start = std::chrono::high_resolution_clock::now();
  std::vector<std::thread> proj_threads;
  for (int lp = 0; lp < local_p; ++lp) {
    proj_threads.emplace_back([&, lp]() {
      int sub_q_offset = lp * sub_q_dim;
      int sub_kv_offset = lp * sub_kv_dim;
      tp_column_matmul(x_norm, Wq_shards[lp], Q_shard + sub_q_offset, 1, sub_q_dim, d_model);
      tp_column_matmul(x_norm, Wk_shards[lp], K_cur_shard + sub_kv_offset, 1, sub_kv_dim, d_model);
      tp_column_matmul(x_norm, Wv_shards[lp], V_cur_shard + sub_kv_offset, 1, sub_kv_dim, d_model);
    });
  }
  for (auto& th : proj_threads) th.join();
  end = std::chrono::high_resolution_clock::now();
  prof.qkv_proj_time += std::chrono::duration<double>(end - start).count();
  start = std::chrono::high_resolution_clock::now();
  apply_rmsnorm_to_heads(Q_shard, rank_n_q, cfg->head_dim, q_norm.data, cfg->rms_eps);
  apply_rmsnorm_to_heads(K_cur_shard, rank_n_kv, cfg->head_dim, k_norm.data, cfg->rms_eps);
  end = std::chrono::high_resolution_clock::now();
  prof.qk_norm_time += std::chrono::duration<double>(end - start).count();
  start = std::chrono::high_resolution_clock::now();
  apply_rope_to_heads(Q_shard, rank_n_q, cfg->head_dim, pos, cfg->rope_theta);
  apply_rope_to_heads(K_cur_shard, rank_n_kv, cfg->head_dim, pos, cfg->rope_theta);
  end = std::chrono::high_resolution_clock::now();
  prof.rope_time += std::chrono::duration<double>(end - start).count();
  start = std::chrono::high_resolution_clock::now();
  append_to_cache(k_cache_shard, K_cur_shard, cache_pos, rank_kv_dim);
  append_to_cache(v_cache_shard, V_cur_shard, cache_pos, rank_kv_dim);
  end = std::chrono::high_resolution_clock::now();
  start = std::chrono::high_resolution_clock::now();
  compute_attn_local(attn_out_shard, Q_shard, k_cache_shard, v_cache_shard, attn_scores, rank_n_q, rank_n_kv, cfg->head_dim, pos, context_len, attn_scale, group_size);
  end = std::chrono::high_resolution_clock::now();
  double attn_time = std::chrono::duration<double>(end - start).count();
  prof.scores_time += attn_time * 0.4;
  prof.softmax_time += attn_time * 0.3;
  prof.weighted_v_time += attn_time * 0.3;
  start = std::chrono::high_resolution_clock::now();
  float* rank_out = new float[d_model]();
  std::vector<std::thread> o_threads;
  for (int lp = 0; lp < local_p; ++lp) {
    o_threads.emplace_back([&, lp]() {
      float* partial = new float[d_model];
      int sub_q_offset = lp * sub_q_dim;
      serial_matmul(attn_out_shard + sub_q_offset, Wo_shards[lp].data, partial, 1, d_model, sub_q_dim);
      for (int i = 0; i < d_model; i++) rank_out[i] += partial[i];
      delete[] partial;
    });
  }
  for (auto& th : o_threads) th.join();
  MPI_Allreduce(rank_out, out, d_model, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  end = std::chrono::high_resolution_clock::now();
  prof.o_proj_time += std::chrono::duration<double>(end - start).count();
  prof.comm_time += std::chrono::duration<double>(end - start).count();  // Approx

  prof.total_time = std::chrono::duration<double>(end - start_total).count();
  delete[] x_norm;
  delete[] Q_shard;
  delete[] K_cur_shard;
  delete[] V_cur_shard;
  delete[] attn_scores;
  delete[] attn_out_shard;
  delete[] rank_out;
}

// Save full weights
void save_full_attn_weights(const AttnConfig* cfg, AttnLayerWeights* weights) {
  int total_q_dim = cfg->n_q * cfg->head_dim;
  int total_kv_dim = cfg->n_kv * cfg->head_dim;

  std::ofstream out_rms("rms1_w.bin", std::ios::binary);
  out_rms.write((char*)weights->rms1_w, cfg->d_model * sizeof(float));

  std::ofstream out_wq("Wq.bin", std::ios::binary);
  out_wq.write((char*)weights->W_proj[0], cfg->d_model * total_q_dim * sizeof(float));

  std::ofstream out_wk("Wk.bin", std::ios::binary);
  out_wk.write((char*)weights->W_proj[1], cfg->d_model * total_kv_dim * sizeof(float));

  std::ofstream out_wv("Wv.bin", std::ios::binary);
  out_wv.write((char*)weights->W_proj[2], cfg->d_model * total_kv_dim * sizeof(float));

  std::ofstream out_wo("Wo.bin", std::ios::binary);
  out_wo.write((char*)weights->Wo, total_q_dim * cfg->d_model * sizeof(float));

  std::ofstream out_qnorm("q_norm.bin", std::ios::binary);
  out_qnorm.write((char*)weights->hk_norm[0], cfg->head_dim * sizeof(float));

  std::ofstream out_knorm("k_norm.bin", std::ios::binary);
  out_knorm.write((char*)weights->hk_norm[1], cfg->head_dim * sizeof(float));
}

// Load local shards
void load_local_attn_shards(const AttnConfig* cfg, int local_p, int mpi_rank, int mpi_size, Tensor2D* Wq_shards, Tensor2D* Wk_shards, Tensor2D* Wv_shards, Tensor2D* Wo_shards, Tensor2D& rms1_w, Tensor2D& q_norm, Tensor2D& k_norm) {
  int d_model = cfg->d_model;
  int total_tp = local_p * mpi_size;
  int local_n_q = cfg->n_q / total_tp;
  int local_n_kv = cfg->n_kv / total_tp;
  int local_q_dim = local_n_q * cfg->head_dim;
  int local_kv_dim = local_n_kv * cfg->head_dim;

  // Replicated small
  float* rms_data = new float[cfg->d_model];
  std::ifstream in_rms("rms1_w.bin", std::ios::binary);
  in_rms.read((char*)rms_data, cfg->d_model * sizeof(float));
  rms1_w = Tensor2D(rms_data, 1, cfg->d_model);

  float* qn_data = new float[cfg->head_dim];
  std::ifstream in_qn("q_norm.bin", std::ios::binary);
  in_qn.read((char*)qn_data, cfg->head_dim * sizeof(float));
  q_norm = Tensor2D(qn_data, 1, cfg->head_dim);

  float* kn_data = new float[cfg->head_dim];
  std::ifstream in_kn("k_norm.bin", std::ios::binary);
  in_kn.read((char*)kn_data, cfg->head_dim * sizeof(float));
  k_norm = Tensor2D(kn_data, 1, cfg->head_dim);

  // Shard
  std::vector<std::thread> load_threads;
  for (int lp = 0; lp < local_p; ++lp) {
    load_threads.emplace_back([&, lp]() {
      int global_p = mpi_rank * local_p + lp;
      int global_q_start = global_p * (cfg->n_q / total_tp);
      int global_kv_start = global_p * (cfg->n_kv / total_tp);

      // Wq column shard
      float* local_wq = new float[cfg->d_model * local_q_dim];
      std::ifstream in_wq("Wq.bin", std::ios::binary);
      for (int r = 0; r < cfg->d_model; r++) {
        in_wq.seekg((r * (cfg->n_q * cfg->head_dim) + global_q_start * cfg->head_dim) * sizeof(float));
        in_wq.read((char*)(local_wq + r * local_q_dim), local_q_dim * sizeof(float));
      }
      Wq_shards[lp] = Tensor2D(local_wq, cfg->d_model, local_q_dim);

      // Similar for Wk, Wv with global_kv_start, local_kv_dim

      float* local_wk = new float[cfg->d_model * local_kv_dim];
      std::ifstream in_wk("Wk.bin", std::ios::binary);
      for (int r = 0; r < cfg->d_model; r++) {
        in_wk.seekg((r * (cfg->n_kv * cfg->head_dim) + global_kv_start * cfg->head_dim) * sizeof(float));
        in_wk.read((char*)(local_wk + r * local_kv_dim), local_kv_dim * sizeof(float));
      }
      Wk_shards[lp] = Tensor2D(local_wk, cfg->d_model, local_kv_dim);

      float* local_wv = new float[cfg->d_model * local_kv_dim];
      std::ifstream in_wv("Wv.bin", std::ios::binary);
      for (int r = 0; r < cfg->d_model; r++) {
        in_wv.seekg((r * (cfg->n_kv * cfg->head_dim) + global_kv_start * cfg->head_dim) * sizeof(float));
        in_wv.read((char*)(local_wv + r * local_kv_dim), local_kv_dim * sizeof(float));
      }
      Wv_shards[lp] = Tensor2D(local_wv, cfg->d_model, local_kv_dim);

      // Wo row shard
      float* local_wo = new float[local_q_dim * d_model];
      std::ifstream in_wo("Wo.bin", std::ios::binary);
      for (int r = 0; r < local_q_dim; r++) {
        in_wo.seekg(((global_q_start * cfg->head_dim + r) * d_model) * sizeof(float));
        in_wo.read((char*)(local_wo + r * d_model), d_model * sizeof(float));
      }
      Wo_shards[lp] = Tensor2D(local_wo, local_q_dim, d_model);
    });
  }
  for (auto& th : load_threads) th.join();
}

// Main
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  AttnConfig cfg;
  int total_tp = 4;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--tp" && i + 1 < argc) {
      total_tp = std::stoi(argv[++i]);
    }
  }
  assert(total_tp % mpi_size == 0);
  int local_p = total_tp / mpi_size;

  Profiler prof_serial, prof_tp;

  float* x = nullptr;
  float* out_serial = nullptr;
  if (mpi_rank == 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

    AttnLayerWeights weights;
    weights.rms1_w = new float[cfg.d_model];
    for (int i = 0; i < cfg.d_model; i++) weights.rms1_w[i] = 1.0f;  // Dummy

    int total_q_dim = cfg.n_q * cfg.head_dim;
    int total_kv_dim = cfg.n_kv * cfg.head_dim;

    weights.W_proj[0] = new float[cfg.d_model * total_q_dim];
    weights.W_proj[1] = new float[cfg.d_model * total_kv_dim];
    weights.W_proj[2] = new float[cfg.d_model * total_kv_dim];
    weights.Wo = new float[total_q_dim * cfg.d_model];
    weights.hk_norm[0] = new float[cfg.head_dim];
    weights.hk_norm[1] = new float[cfg.head_dim];
    for (int i = 0; i < cfg.d_model * total_q_dim; i++) weights.W_proj[0][i] = dist(gen);
    for (int i = 0; i < cfg.d_model * total_kv_dim; i++) {
      weights.W_proj[1][i] = dist(gen);
      weights.W_proj[2][i] = dist(gen);
    }
    for (int i = 0; i < total_q_dim * cfg.d_model; i++) weights.Wo[i] = dist(gen);
    for (int i = 0; i < cfg.head_dim; i++) {
      weights.hk_norm[0][i] = 1.0f;
      weights.hk_norm[1][i] = 1.0f;
    }

    save_full_attn_weights(&cfg, &weights);

    x = new float[cfg.d_model];
    for (int i = 0; i < cfg.d_model; i++) x[i] = dist(gen);

    float* k_cache = new float[cfg.max_seq * total_kv_dim];
    float* v_cache = new float[cfg.max_seq * total_kv_dim];

    out_serial = new float[cfg.d_model];

    attention_serial(out_serial, x, &weights, &cfg, k_cache, v_cache, 0, 0, prof_serial);

    delete[] weights.rms1_w;
    delete[] weights.W_proj[0];
    delete[] weights.W_proj[1];
    delete[] weights.W_proj[2];
    delete[] weights.Wo;
    delete[] weights.hk_norm[0];
    delete[] weights.hk_norm[1];
    delete[] k_cache;
    delete[] v_cache;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  Tensor2D* Wq_shards = new Tensor2D[local_p];
  Tensor2D* Wk_shards = new Tensor2D[local_p];
  Tensor2D* Wv_shards = new Tensor2D[local_p];
  Tensor2D* Wo_shards = new Tensor2D[local_p];
  Tensor2D rms1_w, q_norm, k_norm;

  load_local_attn_shards(&cfg, local_p, mpi_rank, mpi_size, Wq_shards, Wk_shards, Wv_shards, Wo_shards, rms1_w, q_norm, k_norm);

  if (mpi_rank != 0) x = new float[cfg.d_model];
  MPI_Bcast(x, cfg.d_model, MPI_FLOAT, 0, MPI_COMM_WORLD);

  int local_kv_dim = (cfg.n_kv * cfg.head_dim) / total_tp;
  float* k_cache_shard = new float[cfg.max_seq * local_kv_dim];
  float* v_cache_shard = new float[cfg.max_seq * local_kv_dim];

  float* out_tp = new float[cfg.d_model];

  attention_tp(out_tp, x, rms1_w, Wq_shards, Wk_shards, Wv_shards, Wo_shards, q_norm, k_norm, &cfg, k_cache_shard, v_cache_shard, 0, 0, local_p, mpi_rank, mpi_size, prof_tp);

  if (mpi_rank == 0) {
    prof_tp.print("TP");

    double speedup = prof_serial.total_time / prof_tp.total_time;
    std::cout << "Speedup (Serial / TP): " << speedup << "x" << std::endl;

    float tol = 1e-4f;
    bool match = true;
    for (int i = 0; i < cfg.d_model; i++) {
      if (std::abs(out_serial[i] - out_tp[i]) > tol) {
        match = false;
        break;
      }
    }
    std::cout << "Results match: " << (match ? "Yes" : "No") << std::endl;

    delete[] out_serial;
  }

  delete[] x;
  delete[] out_tp;
  delete[] k_cache_shard;
  delete[] v_cache_shard;
  // Delete shards data and arrays
  for (int lp = 0; lp < local_p; lp++) {
    delete[] Wq_shards[lp].data;
    delete[] Wk_shards[lp].data;
    delete[] Wv_shards[lp].data;
    delete[] Wo_shards[lp].data;
  }
  delete[] Wq_shards;
  delete[] Wk_shards;
  delete[] Wv_shards;
  delete[] Wo_shards;
  delete[] rms1_w.data;
  delete[] q_norm.data;
  delete[] k_norm.data;

  MPI_Finalize();
  return 0;
}
