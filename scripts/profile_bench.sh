#!/usr/bin/env bash
#
# Profile benchmark execution using macOS sample tool
#
# Usage: ./profile_bench.sh [OPTIONS]
#
# Options:
#   -d, --duration SECONDS    Sampling duration (default: 300)
#   -i, --interval MS         Sampling interval in milliseconds (default: 1)
#   -o, --output FILE         Output file path (default: eval/profile.txt)
#   -b, --binary NAME         Binary name pattern (default: algorithms)
#   -k, --kill-after-profile  Kill benchmark immediately after profiling completes
#   -h, --help                Show this help message
#

set -e  # Exit on error

# Default values
DURATION=300
INTERVAL=1
OUTPUT_FILE="eval/profile.txt"
BINARY_PATTERN="algorithms"
KILL_AFTER_PROFILE=false
BENCH_PID=""

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -b|--binary)
            BINARY_PATTERN="$2"
            shift 2
            ;;
        -k|--kill-after-profile)
            KILL_AFTER_PROFILE=true
            shift
            ;;
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}" >&2
            echo "Run with --help for usage information" >&2
            exit 1
            ;;
    esac
done

# Function to find the benchmark binary
find_benchmark_binary() {
    local pattern="$1"
    local binary_path

    # Build release benchmarks if needed
    if [ ! -d "target/release/deps" ]; then
        echo -e "${YELLOW}Building benchmarks...${NC}"
        cargo build --benches --release
    fi

    # Find the most recent matching executable
    binary_path=$(find target/release/deps -type f -perm +111 -name "${pattern}-*" | head -n 1)

    if [ -z "$binary_path" ]; then
        echo -e "${RED}Error: Could not find benchmark binary matching '${pattern}'${NC}" >&2
        echo "Available binaries:" >&2
        find target/release/deps -type f -perm +111 -name "*-*" | xargs -n1 basename >&2
        exit 1
    fi

    echo "$binary_path"
}

# Function to check if sample command is available
check_sample_available() {
    if ! command -v sample &> /dev/null; then
        echo -e "${RED}Error: 'sample' command not found${NC}" >&2
        echo "This script requires the macOS sample tool" >&2
        exit 1
    fi
}

# Cleanup function to kill benchmark on exit
cleanup() {
    if [ -n "$BENCH_PID" ] && kill -0 "$BENCH_PID" 2>/dev/null; then
        echo ""
        echo -e "${YELLOW}Terminating benchmark process (PID: ${BENCH_PID})...${NC}"
        kill "$BENCH_PID" 2>/dev/null || true
        wait "$BENCH_PID" 2>/dev/null || true
        echo -e "${GREEN}✓ Benchmark terminated${NC}"
    fi
}

# Set up trap to handle interrupts
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    echo -e "${GREEN}=== Benchmark Profiling Setup ===${NC}"
    echo "Duration: ${DURATION}s"
    echo "Interval: ${INTERVAL}ms"
    echo "Output: ${OUTPUT_FILE}"
    echo ""

    # Check prerequisites
    check_sample_available

    # Find the benchmark binary
    echo -e "${YELLOW}Finding benchmark binary...${NC}"
    BINARY_PATH=$(find_benchmark_binary "$BINARY_PATTERN")
    echo "Found: ${BINARY_PATH}"
    echo ""

    # Create output directory if needed
    OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
    mkdir -p "$OUTPUT_DIR"

    # Run benchmark in background and capture PID
    echo -e "${YELLOW}Starting benchmark...${NC}"
    "$BINARY_PATH" --bench > /dev/null 2>&1 &
    BENCH_PID=$!
    echo "Benchmark PID: ${BENCH_PID}"

    # Wait a moment for the benchmark to start
    sleep 0.5

    # Check if process is still running
    if ! kill -0 "$BENCH_PID" 2>/dev/null; then
        echo -e "${RED}Error: Benchmark process died immediately${NC}" >&2
        exit 1
    fi

    # Run sample on the benchmark process
    echo -e "${YELLOW}Profiling for ${DURATION} seconds...${NC}"
    if sample "$BENCH_PID" "$DURATION" "$INTERVAL" -file "$OUTPUT_FILE"; then
        echo -e "${GREEN}✓ Profile saved to: ${OUTPUT_FILE}${NC}"
    else
        echo -e "${RED}✗ Profiling failed${NC}" >&2
        kill "$BENCH_PID" 2>/dev/null || true
        exit 1
    fi

    # Handle benchmark completion based on flag
    if [ "$KILL_AFTER_PROFILE" = true ]; then
        echo -e "${YELLOW}Killing benchmark process...${NC}"
        kill "$BENCH_PID" 2>/dev/null || true
        wait "$BENCH_PID" 2>/dev/null || true
        echo -e "${GREEN}✓ Benchmark terminated${NC}"
    else
        # Wait for benchmark to complete (Ctrl+C will trigger cleanup trap)
        echo -e "${YELLOW}Waiting for benchmark to complete (press Ctrl+C to stop)...${NC}"
        wait "$BENCH_PID" 2>/dev/null || true
    fi

    echo ""
    echo -e "${GREEN}=== Profiling Complete ===${NC}"
    echo "Profile data: ${OUTPUT_FILE}"
    echo "Lines: $(wc -l < "$OUTPUT_FILE")"

    # Clear BENCH_PID so cleanup doesn't try to kill it again
    BENCH_PID=""
}

# Run main function
main
