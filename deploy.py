# deploy.py
import os
import argparse
import subprocess
import sys

def check_requirements():
    """Check if all required packages are installed"""
    required = [
        "fastapi", "uvicorn", "pydantic", 
        "langchain", "openai", "configparser"
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
    dirs = ["logs", "artifacts", "conversations", "assessments"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Created necessary directories")

def deploy(host, port, reload):
    """Deploy the API using uvicorn"""
    print(f"Deploying API on {host}:{port}")
    
    # Convert relative import paths to absolute paths
    module_path = "api.server:app"
    
    # Start uvicorn
    cmd = ["uvicorn", module_path, "--host", host, "--port", str(port)]
    if reload:
        cmd.append("--reload")
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

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
        deploy(args.host, args.port, args.reload)
    else:
        print("Requirements check passed. Ready to deploy.")

if __name__ == "__main__":
    main()