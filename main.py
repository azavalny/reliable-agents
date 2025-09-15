"""Main entry point for the Reliable Agents system."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Run the main application."""
    import subprocess
    
    print("🤖 Reliable Agents - High Precision AI System")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit available")
    except ImportError:
        print("❌ Streamlit not installed. Run: pip install -r requirements.txt")
        return
    
    # Run streamlit app
    print("🚀 Starting web interface...")
    print("Open your browser to: http://localhost:8501")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Reliable Agents system")

if __name__ == "__main__":
    main()
