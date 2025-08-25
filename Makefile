# Makefile for compiling the MoE tensor-parallel prototype
# Compatible with Linux and macOS (OSX)
# Automatically detects MPI usage and compiles accordingly

CXX = g++
MPICXX = mpic++
CXXFLAGS = -std=c++17 -pthread -O2

# Define all targets
TARGETS = moe_test_0 moe_test_1 moe_test_2 attn_test_1 attn_test_2

# Files that use MPI
MPI_TARGETS = moe_test_1 moe_test_2 attn_test_1 attn_test_2

all: $(TARGETS)

# Default rule for non-MPI targets
%: %.cc
	@if echo $(MPI_TARGETS) | grep -q $@; then \
		echo "Compiling $@ with MPI support..."; \
		$(MPICXX) $(CXXFLAGS) -o $@ $<; \
	else \
		echo "Compiling $@ without MPI..."; \
		$(CXX) $(CXXFLAGS) -o $@ $<; \
	fi

clean:
	rm -f $(TARGETS)
