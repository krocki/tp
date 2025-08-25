# Makefile for compiling the MoE tensor-parallel prototype
# Compatible with Linux and macOS (OSX)
# Assumes g++ is available (install via Homebrew on macOS if needed)

CXX = g++
CXXFLAGS = -std=c++17 -pthread -O2  # Add -O2 for optimization
all: moe_test_0 moe_test_1

%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

moe_test_1: moe_test_1.cc
	mpic++ $(CXXFLAGS) -o $@ $<

clean:
	rm -f moe_test_0 moe_test_1
