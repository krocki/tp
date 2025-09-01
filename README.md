# Tensor Parallel Implementations for Transformer Blocks

This repository contains tensor parallel implementations of key transformer components: Mixture of Experts (MOE) and multi-head attention layers. The implementations demonstrate different parallelization strategies and support both single-token and batched processing.

## Overview

The codebase provides several test programs that implement tensor parallel versions of transformer blocks:

### MOE (Mixture of Experts) Implementations
- **`moe_test_0.cc`**: Local tensor parallelism (no MPI), single token processing
- **`moe_test_1.cc`**: MPI-based tensor parallelism, single token processing  
- **`moe_test_2.cc`**: MPI-based tensor parallelism, batch processing support

### Attention Implementations
- **`attn_test_1.cc`**: MPI-based tensor parallel attention, single token processing
- **`attn_test_2.cc`**: MPI-based tensor parallel attention, batch processing support

## Architecture

### Tensor Parallelism Strategy

All implementations use **column-wise parallelism** for the feedforward layers:
- Gate/Up projections: Split along output dimension (d_ff axis) 
- Down projection: Split along input dimension, requires AllReduce
- Router: Replicated across all ranks

For attention layers:
- Q/K/V projections: Split along head dimension
- Output projection: Split along input dimension, requires AllReduce
- Supports both head splitting and head replication for different tp/n_kv ratios

### Configuration

Default model configuration:
```cpp
struct QwenConfig {
  int d_model = 2048;    // Model hidden dimension
  int d_ff = 768;        // Feed-forward dimension per expert
  int n_experts = 128;   // Total number of experts
  int top_k = 8;         // Number of experts activated per token
};

struct AttnConfig {
  int d_model = 2048;    // Model hidden dimension  
  int n_q = 32;          // Number of query heads
  int n_kv = 4;          // Number of key-value heads (GQA)
  int head_dim = 64;     // Dimension per attention head
  int max_seq_len = 2048; // Maximum sequence length
};
```

## Building

The project uses a unified Makefile that automatically detects MPI requirements:

```bash
# Build all targets
make all

# Build specific targets
make moe_test_0      # No MPI required
make moe_test_1      # Requires MPI
make moe_test_2      # Requires MPI  
make attn_test_1     # Requires MPI
make attn_test_2     # Requires MPI

# Clean build artifacts
make clean
```

### Requirements
- C++17 compatible compiler
- MPI implementation (OpenMPI, MPICH, etc.) for MPI-enabled targets
- POSIX threads support

## Usage

### Command Line Arguments
All test programs support the following arguments:
```bash
./program [--tp <tensor_parallelism>] [--batch <batch_size>]
```

- `--tp`: Degree of tensor parallelism (default: 1)
- `--batch`: Batch size for processing (default: 1)

### Examples

#### MOE Tests
```bash
# Local tensor parallel MOE (no MPI)
./moe_test_0

# MPI tensor parallel MOE, single token
mpirun -np 2 ./moe_test_1 --tp 2 --batch 1

# MPI tensor parallel MOE, batch processing  
mpirun -np 4 ./moe_test_2 --tp 4 --batch 8
```

#### Attention Tests
```bash
# MPI tensor parallel attention, single token
mpirun -np 2 ./attn_test_1 --tp 2 --batch 1

# MPI tensor parallel attention, batch processing
mpirun -np 4 ./attn_test_2 --tp 4 --batch 16
```

### Constraints

#### Tensor Parallelism Constraints
- **MOE**: `tp` must divide `d_ff` (768), so valid values are: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32...
- **Attention**: `tp` constraints depend on head configuration and support both head splitting and replication

#### MPI Constraints  
- Number of MPI ranks must equal `tp` (each rank handles exactly one shard)
- `np` (MPI ranks) cannot exceed `tp`

## Testing

### Automated Test Suites

#### MOE Testing
```bash
# Test all valid MOE configurations
./test_moe_configs.sh

# Key test scenarios:
# - tp=1,2,3,4,6,8,12,16,24,32 
# - batch=1,2,4,8,16,32
# - Various np/tp combinations
```

#### Attention Testing  
```bash
# Test all valid attention configurations
./test_all_configs.sh

# Test only valid np=tp configurations
./test_valid_configs.sh
```

### Performance Analysis
All tests provide detailed performance breakdowns:
- Component-wise timing (router, projections, communication, etc.)
- Speedup measurements vs serial execution
- Correctness validation (numerical comparison with reference implementation)

### Expected Performance
- **Good scaling**: tp=2 should achieve ~1.7-1.9x speedup
- **Excellent scaling**: tp=4 should achieve ~3.5-4.0x speedup  
- **Strong scaling**: tp=8 should achieve ~6.0-7.5x speedup

Communication overhead becomes more significant at higher parallelism degrees.

## Weight Management

### Caching System
Recent versions (moe_test_2, attn_test_2) include intelligent weight caching:
- Weights are saved to `weights/` subdirectory on first run
- Subsequent runs load cached weights instead of regenerating
- Significantly reduces startup time for repeated testing

### Weight Files
- **MOE**: `router.bin` + `expert{0-127}_{Wg,Wu,Wd}.bin` (385 files total)  
- **Attention**: `rms1_w.bin`, `{Wq,Wk,Wv,Wo}.bin`, `{q,k}_norm.bin` (7 files total)

## Implementation Details

### Memory Layout
- All matrices stored in row-major format
- Tensor shards maintain contiguous memory layout for efficient access
- Separate allocation for each MPI rank's weight shards

### Communication Patterns
- **AllReduce**: Used after row-parallel operations (down projection, output projection)
- **Broadcast**: Used for input distribution from rank 0
- **Barrier**: Used for synchronization during weight file I/O

### Numerical Precision
- All computations use single-precision floating point (float32)
- Correctness validation uses absolute tolerance of 1e-4
- Timing measurements use high-resolution chrono timers

## Code Structure

### Common Components (`common.h`)
- Shared data structures (configurations, profilers, tensor representations)
- Utility functions (matrix multiplication, activations, normalization)
- Command line argument parsing

### Modular Design
Each implementation follows a consistent pattern:
1. **Serial Reference**: Baseline implementation for correctness validation
2. **Tensor Parallel**: Optimized implementation with parallelism
3. **Performance Comparison**: Detailed timing and speedup analysis
4. **Correctness Validation**: Numerical comparison between implementations

## Future Extensions

This modular design enables easy integration into complete transformer models by:
1. Standardizing the tensor parallel interface across components
2. Providing unified MPI rank and parallelism degree handling
3. Establishing consistent weight sharding strategies
4. Creating reusable profiling and validation frameworks

The implementations can serve as building blocks for larger transformer models while maintaining the proven scaling characteristics demonstrated in the individual tests.