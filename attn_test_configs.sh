#!/bin/bash

# Test script for valid tensor parallel configurations only
# Tests cases where np = tp (each rank handles exactly one shard)

echo "=== Valid Tensor Parallel Configuration Testing ==="
echo "Testing only cases where np = tp (each MPI rank handles one shard)"
echo "Current config: n_q=32, n_kv=4"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
total_tests=0
passed_tests=0
failed_tests=0
skipped_tests=0

# Function to run a single test
run_test() {
    local np=$1
    local tp=$2
    local batch=$3
    
    total_tests=$((total_tests + 1))
    
    # Validate configuration
    if [ $((32 % tp)) -ne 0 ]; then
        echo -e "  ${YELLOW}SKIP${NC}: n_q=32 not divisible by tp=$tp"
        skipped_tests=$((skipped_tests + 1))
        return
    fi
    
    if [ $tp -le 4 ]; then
        if [ $((4 % tp)) -ne 0 ]; then
            echo -e "  ${YELLOW}SKIP${NC}: n_kv=4 not divisible by tp=$tp (split case)"
            skipped_tests=$((skipped_tests + 1))
            return
        fi
    else
        if [ $((tp % 4)) -ne 0 ]; then
            echo -e "  ${YELLOW}SKIP${NC}: tp=$tp not divisible by n_kv=4 (replicate case)"
            skipped_tests=$((skipped_tests + 1))
            return
        fi
    fi
    
    # Run the test with timeout and capture full output
    local output
    output=$(timeout 30s mpirun -np $np ./attn_test_2 --tp $tp --batch $batch 2>/dev/null)
    
    local result=$(echo "$output" | grep "Results match" | tail -1)
    local speedup=$(echo "$output" | grep "Speedup" | tail -1 | sed 's/.*Speedup.*: \([0-9.]*\)x.*/\1/')
    
    if [[ $result == *"Results match: Yes"* ]]; then
        if [[ -n "$speedup" && "$speedup" != "" ]]; then
            local color="${GREEN}"
            # Color code based on speedup
            if (( $(echo "$speedup > 2.0" | bc -l) )); then
                color="${BLUE}"  # Excellent speedup
            fi
            echo -e "  ${color}PASS${NC}: np=$np tp=$tp batch=$batch (${speedup}x speedup)"
        else
            echo -e "  ${GREEN}PASS${NC}: np=$np tp=$tp batch=$batch"
        fi
        passed_tests=$((passed_tests + 1))
    elif [[ $result == *"Results match: No"* ]]; then
        if [[ -n "$speedup" && "$speedup" != "" ]]; then
            echo -e "  ${RED}FAIL${NC}: np=$np tp=$tp batch=$batch (${speedup}x speedup)"
        else
            echo -e "  ${RED}FAIL${NC}: np=$np tp=$tp batch=$batch"
        fi
        failed_tests=$((failed_tests + 1))
    else
        echo -e "  ${RED}ERROR${NC}: np=$np tp=$tp batch=$batch (timeout/crash)"
        failed_tests=$((failed_tests + 1))
    fi
}

# Make sure the binary is compiled
echo "Compiling attn_test_2..."
make attn_test_2 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to compile attn_test_2${NC}"
    exit 1
fi

echo "Starting tests..."
echo ""

# Test ranges - only valid configurations where np = tp
TP_VALUES=(1 2 4 8 16 32)
BATCH_VALUES=(1 2 4 8 16 32)

# Test all combinations with np = tp
for tp in "${TP_VALUES[@]}"; do
    echo "=== Testing tp=$tp (np=$tp) ==="
    
    for batch in "${BATCH_VALUES[@]}"; do
        # Only test with np = tp (valid configuration)
        run_test $tp $tp $batch
    done
    echo ""
done

# Print summary
echo "=== TEST SUMMARY ==="
echo "Total tests: $total_tests"
echo -e "Passed: ${GREEN}$passed_tests${NC}"
echo -e "Failed: ${RED}$failed_tests${NC}" 
echo -e "Skipped: ${YELLOW}$skipped_tests${NC}"
echo ""

if [ $failed_tests -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    echo "The tensor parallel implementation supports:"
    echo "• tp values: 1, 2, 4 (split KV heads), 8, 16, 32 (replicate KV heads)"
    echo "• batch sizes: 1-32 tokens"
    echo "• Requirement: np = tp (each MPI rank handles one shard)"
    exit 0
else
    echo -e "${RED}$failed_tests tests failed.${NC}"
    exit 1
fi