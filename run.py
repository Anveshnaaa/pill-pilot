#!/usr/bin/env python3
"""
Medicine Inventory Dashboard - Startup Script
Run this script to start the application
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def start_app():
    """Start the Flask application"""
    print("Starting Medicine Inventory Dashboard...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("📊 Upload the sample_inventory.csv file to see the dashboard in action!")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")

if __name__ == "__main__":
    print("🏥 Medicine Inventory Dashboard")
    print("=" * 40)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        sys.exit(1)
    
    # Install requirements
    if install_requirements():
        start_app()
    else:
        print("❌ Failed to install requirements. Please install manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
