#!/bin/bash
# Test Coverage Report Generator for x.titan

# Ensure we are in the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

echo "🚀 Starting Test Coverage Report..."

# Create reports directory if it doesn't exist
mkdir -p coverage_reports

# Run pytest with coverage
# We focus on the core modules: execution, data, models, and config
./venv/bin/pytest --cov=execution \
       --cov=data \
       --cov=models \
       --cov=config \
       --cov-report=term-missing \
       --cov-report=html:coverage_reports/html \
       --cov-report=xml:coverage_reports/coverage.xml \
       tests/

echo ""
echo "✅ Coverage report generated in coverage_reports/html/index.html"
echo "✅ XML report available at coverage_reports/coverage.xml"
