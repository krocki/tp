#!/bin/bash

# Comprehensive test script for moe_test_2 tensor parallel configurations
# Tests MOE (Mixture of Experts) with various tp, batch, and np configurations

echo "=== MOE Tensor Parallel Configuration Testing ==="
echo "Testing moe_test_2 with various tp, batch, and np configurations"
echo "Current config: d_model=2048, d_ff=768, n_experts=128, top_k=8"
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
    # d_ff=768 must be divisible by tp
    if [ $((768 % tp)) -ne 0 ]; then
        echo -e "  ${YELLOW}SKIP${NC}: d_ff=768 not divisible by tp=$tp"
        skipped_tests=$((skipped_tests + 1))
        return
    fi
    
    # tp must be divisible by np (each rank handles tp/np shards)
    if [ $((tp % np)) -ne 0 ]; then
        echo -e "  ${YELLOW}SKIP${NC}: tp=$tp not divisible by np=$np"
        skipped_tests=$((skipped_tests + 1))
        return
    fi
    
    # np cannot exceed tp (each rank handles at least one shard)
    if [ $np -gt $tp ]; then
        echo -e "  ${YELLOW}SKIP${NC}: np=$np cannot exceed tp=$tp"
        skipped_tests=$((skipped_tests + 1))
        return
    fi
    
    # Run the test with timeout and capture full output
    local output
    output=$(timeout 60s mpirun -np $np ./moe_test_2 --tp $tp --batch $batch 2>/dev/null)
    
    local result=$(echo "$output" | grep "Results match" | tail -1)
    local speedup=$(echo "$output" | grep "Speedup" | tail -1 | sed 's/.*Speedup.*: \([0-9.]*\)x.*/\1/')
    
    if [[ $result == *"Results match: Yes"* ]]; then
        if [[ -n "$speedup" && "$speedup" != "" ]]; then
            local color="${GREEN}"
            # Color code based on speedup
            if (( $(echo "$speedup > 3.0" | bc -l) )); then
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
echo "Compiling moe_test_2..."
make moe_test_2 > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to compile moe_test_2${NC}"
    exit 1
fi

echo "Starting tests..."
echo ""

# Test ranges - valid TP values are divisors of d_ff=768
# 768 = 2^8 * 3, so divisors include: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768
TP_VALUES=(1 2 3 4 6 8 12 16 24 32)
BATCH_VALUES=(1 2 4 8 16 32)

# Test configurations where np <= tp
for tp in "${TP_VALUES[@]}"; do
    echo "=== Testing tp=$tp ==="
    
    for batch in "${BATCH_VALUES[@]}"; do
        for np in $(seq 1 $tp); do
            # Skip if tp % np != 0
            if [ $((tp % np)) -ne 0 ]; then
                continue
            fi
            
            # Skip very large np values to avoid overwhelming the system
            if [ $np -gt 8 ]; then
                continue
            fi
            
            run_test $np $tp $batch
        done
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
    echo "The MOE tensor parallel implementation supports:"
    echo "• tp values: divisors of d_ff=768 (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, etc.)"
    echo "• batch sizes: 1-32 tokens"
    echo "• np values: must divide tp evenly (each rank handles tp/np shards)"
    exit 0
else
    echo -e "${RED}$failed_tests tests failed.${NC}"
    exit 1
fi