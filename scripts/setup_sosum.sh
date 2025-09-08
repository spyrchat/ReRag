#!/bin/bash
# SOSum Dataset Quick Setup Script
# This script downloads and sets up the SOSum dataset for ingestion

set -e

echo "🔧 SOSum Dataset Setup"
echo "======================"

# Configuration
SOSUM_DIR="datasets/sosum"
GITHUB_URL="https://github.com/BonanKou/SOSum-A-Dataset-of-Extractive-Summaries-of-Stack-Overflow-Posts-and-labeling-tools.git"

# Create datasets directory
echo "Creating datasets directory..."
mkdir -p datasets/

# Download SOSum dataset
if [ ! -d "$SOSUM_DIR" ]; then
    echo "Downloading SOSum dataset..."
    git clone "$GITHUB_URL" "$SOSUM_DIR"
else
    echo "SOSum dataset already exists at $SOSUM_DIR"
fi

# Verify files exist
echo "Verifying dataset files..."
if [ -f "$SOSUM_DIR/data/question.csv" ] && [ -f "$SOSUM_DIR/data/answer.csv" ]; then
    echo "Found question.csv and answer.csv"
    
    # Show file stats
    echo "Dataset statistics:"
    echo "   Questions: $(tail -n +2 $SOSUM_DIR/data/question.csv | wc -l) rows"
    echo "   Answers: $(tail -n +2 $SOSUM_DIR/data/answer.csv | wc -l) rows"
else
    echo "Required CSV files not found in $SOSUM_DIR/data/"
    exit 1
fi

echo ""
echo "   Ready for ingestion! Run:"
echo "   # Test the adapter"
echo "   python examples/ingest_sosum_example.py"
echo ""
echo "   # Dry run ingestion"
echo "   python bin/ingest.py ingest stackoverflow $SOSUM_DIR --dry-run --max-docs 10"
echo ""
echo "   # Canary ingestion (safe test)"
echo "   python bin/ingest.py ingest stackoverflow $SOSUM_DIR --canary --max-docs 100"
echo ""
echo "   # Full ingestion"
echo "   python bin/ingest.py ingest stackoverflow $SOSUM_DIR"
echo ""
echo "   # Evaluate retrieval"
echo "   python bin/ingest.py evaluate stackoverflow $SOSUM_DIR --output-dir results/sosum/"
