# deploy.py
import os
import argparse
import subprocess
import sys
from pathlib import Path
import uvicorn
from config.settings import ConfigManager

# Import the app from server.py
from api import server

def check_requirements():
    """Check if all required packages are installed"""
    required = [
        "fastapi", "uvicorn", "pydantic", 
        "langchain", "openai"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        install = input("Install missing packages? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        else:
            print("Please install the required packages and try again.")
            sys.exit(1)

def ensure_directories():
    """Ensure all required directories exist"""
    try:
        config = ConfigManager()
        
        # Get directories from config
        dirs = [
            config.system.conversation_directory,
            config.system.working_directory,
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            print(f"Created directory: {d}")
    except Exception as e:
        print(f"Error setting up directories: {e}")
        print("Creating default directories instead...")
        default_dirs = ["conversations", "working_dir"]
        for d in default_dirs:
            os.makedirs(d, exist_ok=True)
            print(f"Created directory: {d}")

def main():
    parser = argparse.ArgumentParser(description="Deploy Dr.Hyper API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--check", action="store_true", help="Check requirements only")
    
    args = parser.parse_args()
    
    print("Checking requirements...")
    check_requirements()
    
    print("Ensuring directories exist...")
    ensure_directories()
    
    if not args.check:
        print(f"Deploying API on {args.host}:{args.port}")
        uvicorn.run("api.server:app", host=args.host, port=args.port, reload=args.reload)
    else:
        print("Requirements check passed. Ready to deploy.")

if __name__ == "__main__":
    main()