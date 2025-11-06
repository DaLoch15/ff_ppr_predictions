#!/bin/bash
# Test Runner Script for Fantasy Football Projections

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       Fantasy Football Projections - Test Suite             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Python version:"
python3 --version
echo ""

# Function to run tests with a specific class
run_test_class() {
    local class_name=$1
    echo -e "${YELLOW}Running ${class_name}...${NC}"
    python3 -m unittest tests.test_pipeline.${class_name} -v 2>&1 | tail -5
    echo ""
}

# Parse command line arguments
case "$1" in
    "all")
        echo "Running all tests..."
        python3 tests/test_pipeline.py
        ;;
    "db"|"database")
        run_test_class "TestDatabaseManager"
        ;;
    "fetcher"|"data")
        run_test_class "TestDataFetcher"
        ;;
    "validator")
        run_test_class "TestDataValidator"
        ;;
    "ml"|"pipeline")
        run_test_class "TestMLPipeline"
        ;;
    "prod"|"production")
        run_test_class "TestProductionSystem"
        ;;
    "quick")
        echo "Running quick tests (DB, Validator)..."
        python3 -m unittest tests.test_pipeline.TestDatabaseManager tests.test_pipeline.TestDataValidator -v
        ;;
    "coverage")
        echo "Running tests with coverage..."
        if ! command -v coverage &> /dev/null; then
            echo -e "${RED}Coverage not installed. Run: pip install coverage${NC}"
            exit 1
        fi
        coverage run -m unittest discover tests
        coverage report
        coverage html
        echo -e "${GREEN}HTML coverage report generated in htmlcov/index.html${NC}"
        ;;
    *)
        echo "Usage: ./run_tests.sh [option]"
        echo ""
        echo "Options:"
        echo "  all          Run all tests (default)"
        echo "  db           Run database tests only"
        echo "  fetcher      Run data fetcher tests only"
        echo "  validator    Run data validator tests only"
        echo "  ml           Run ML pipeline tests only"
        echo "  prod         Run production system tests only"
        echo "  quick        Run quick tests (DB + Validator)"
        echo "  coverage     Run tests with coverage report"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh all"
        echo "  ./run_tests.sh db"
        echo "  ./run_tests.sh coverage"
        ;;
esac
