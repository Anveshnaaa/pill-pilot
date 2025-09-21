#!/usr/bin/env python3
"""
PillPilot Setup Script
Automated setup and installation for PillPilot Medicine Inventory Management System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print the PillPilot banner"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║  🚀 PillPilot - Medicine Inventory Management System        ║
    ║                                                              ║
    ║  Automated Setup Script v1.0                                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")

def check_pip():
    """Check if pip is available"""
    print("🔍 Checking pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
    except subprocess.CalledProcessError:
        print("❌ Error: pip is not available")
        print("   Please install pip first: https://pip.pypa.io/en/stable/installation/")
        sys.exit(1)

def create_virtual_environment():
    """Create virtual environment"""
    print("🔧 Creating virtual environment...")
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("⚠️  Virtual environment already exists")
        response = input("   Do you want to recreate it? (y/N): ").lower()
        if response == 'y':
            import shutil
            shutil.rmtree(venv_path)
        else:
            print("✅ Using existing virtual environment")
            return
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        sys.exit(1)

def get_activation_command():
    """Get the correct activation command for the platform"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing requirements...")
    
    # Get the correct pip path
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    try:
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        print("   Please check that requirements.txt exists and try again")
        sys.exit(1)

def create_env_file():
    """Create .env file if it doesn't exist"""
    print("🔧 Creating environment configuration...")
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    env_content = """# PillPilot Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-change-this-in-production
ML_ENABLED=True
MODEL_UPDATE_INTERVAL=3600
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created")
    except Exception as e:
        print(f"⚠️  Warning: Could not create .env file: {e}")

def create_run_script():
    """Create platform-specific run script"""
    print("🔧 Creating run script...")
    
    if platform.system() == "Windows":
        script_content = """@echo off
echo Starting PillPilot...
call venv\\Scripts\\activate
python app.py
pause
"""
        script_path = "run.bat"
    else:
        script_content = """#!/bin/bash
echo "Starting PillPilot..."
source venv/bin/activate
python app.py
"""
        script_path = "run.sh"
    
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        if platform.system() != "Windows":
            os.chmod(script_path, 0o755)
        
        print(f"✅ Run script created: {script_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create run script: {e}")

def print_success_message():
    """Print success message with next steps"""
    activation_cmd = get_activation_command()
    
    print("""
    🎉 Setup completed successfully!
    
    Next steps:
    
    1. Activate the virtual environment:
       {activation_cmd}
    
    2. Start the application:
       python app.py
       
       Or use the run script:
       {run_script}
    
    3. Open your browser and go to:
       http://localhost:5000
    
    4. Upload your CSV file to get started!
    
    📚 For more information, see README.md
    🐛 For issues, check the troubleshooting section
    """.format(
        activation_cmd=activation_cmd,
        run_script="run.bat" if platform.system() == "Windows" else "./run.sh"
    ))

def main():
    """Main setup function"""
    print_banner()
    
    try:
        check_python_version()
        check_pip()
        create_virtual_environment()
        install_requirements()
        create_env_file()
        create_run_script()
        print_success_message()
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

