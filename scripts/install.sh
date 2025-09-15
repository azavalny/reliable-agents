#!/bin/bash

echo "🤖 Reliable Agents - Installation Script"
echo "========================================"

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Creating data directories..."
mkdir -p data/samples
mkdir -p models

echo ""
echo "Setup complete!"
echo ""
echo "To run the system:"
echo "  python main.py"
echo ""
echo "Or using the script:"
echo "  python scripts/run.py"
echo ""
echo "Or directly:"
echo "  streamlit run ui/app.py"
echo ""
echo "Don't forget to set your OPENAI_API_KEY environment variable!"
