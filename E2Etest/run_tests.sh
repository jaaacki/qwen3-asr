#!/bin/bash
# Run E2E tests for Qwen3-ASR with optional server management

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
WITH_SERVER=false
SKIP_SMOKE=false
MARKERS=""
VERBOSE="-v"
COVERAGE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-server)
            WITH_SERVER=true
            shift
            ;;
        --skip-smoke)
            SKIP_SMOKE=true
            shift
            ;;
        --http-only)
            MARKERS="-k http"
            shift
            ;;
        --websocket-only)
            MARKERS="-k websocket"
            shift
            ;;
        --performance)
            MARKERS="-m performance"
            shift
            ;;
        --fast)
            MARKERS="-m 'not slow'"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --with-server      Start/stop Docker container for tests"
            echo "  --skip-smoke       Skip smoke tests"
            echo "  --http-only        Run only HTTP tests"
            echo "  --websocket-only   Run only WebSocket tests"
            echo "  --performance      Run only performance tests"
            echo "  --fast             Skip slow tests"
            echo "  --coverage         Generate coverage report"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --with-server           # Full test run with server"
            echo "  $0 --fast                  # Quick smoke tests"
            echo "  $0 --http-only --fast      # HTTP smoke tests only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Qwen3-ASR E2E Test Runner"
echo "========================================"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Check/install dependencies
echo "Checking dependencies..."
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    pip install -q -r E2Etest/requirements.txt
fi
echo -e "${GREEN}Dependencies OK${NC}"
echo ""

# Start server if requested
if [ "$WITH_SERVER" = true ]; then
    echo "Starting server..."
    docker compose up -d --build

    echo "Waiting for server to be healthy..."
    for i in {1..60}; do
        if curl -s http://localhost:8100/health >/dev/null 2>&1; then
            echo -e "${GREEN}Server is ready!${NC}"
            break
        fi
        echo -n "."
        sleep 2
    done

    if ! curl -s http://localhost:8100/health >/dev/null 2>&1; then
        echo -e "${RED}Server failed to start${NC}"
        docker compose logs --tail=50
        exit 1
    fi
    echo ""
fi

# Verify server is running
if ! curl -s http://localhost:8100/health >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Server not responding at localhost:8100${NC}"
    echo "Tests will be skipped. Start server with: docker compose up -d"
    echo "Or use --with-server flag to auto-start"
    echo ""
fi

# Generate test audio if needed
if [ ! -f "E2Etest/data/audio/short_5s.wav" ]; then
    echo "Generating test audio files..."
    python3 << 'EOF'
import sys
sys.path.insert(0, 'E2Etest')
from utils.audio import generate_test_audio_files
generate_test_audio_files('E2Etest/data/audio')
EOF
    echo ""
fi

# Build pytest command
PYTEST_CMD="pytest E2Etest/ $VERBOSE"

if [ "$SKIP_SMOKE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not smoke'"
fi

if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html:E2Etest/htmlcov"
fi

# Run tests
echo "Running tests..."
echo "Command: $PYTEST_CMD"
echo ""
echo "========================================"

set +e
eval $PYTEST_CMD
TEST_EXIT=$?
set -e

echo "========================================"

# Stop server if we started it
if [ "$WITH_SERVER" = true ]; then
    echo ""
    echo "Stopping server..."
    docker compose down
fi

# Summary
echo ""
if [ $TEST_EXIT -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed (exit code: $TEST_EXIT)${NC}"
fi

exit $TEST_EXIT
