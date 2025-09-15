"""Startup script for the Reliable Agents system."""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed."""
    try:
        import streamlit
        import openai
        import torch
        import sklearn
        import xgboost
        import pandas
        import numpy
        import PIL
        import matplotlib
        import cv2
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if OpenAI API key is configured."""
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key is configured")
        return True
    else:
        print("⚠️ OpenAI API key not found")
        print("Set the OPENAI_API_KEY environment variable or create a .env file")
        print("Some features will be limited without the API key")
        return False

def create_directories():
    """Create necessary directories."""
    dirs = ['data', 'data/samples', 'models']
    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True)
    print("✅ Created necessary directories")

def run_streamlit():
    """Run the Streamlit application."""
    print("🚀 Starting Reliable Agents system...")
    print("Open your browser to: http://localhost:8501")
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])

def main():
    """Main startup function."""
    print("🤖 Reliable Agents - High Precision AI System")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check API key
    check_api_key()
    
    # Create directories
    create_directories()
    
    # Run the system
    try:
        run_streamlit()
    except KeyboardInterrupt:
        print("\n👋 Shutting down Reliable Agents system")
    except Exception as e:
        print(f"❌ Error running system: {e}")

if __name__ == "__main__":
    main()
