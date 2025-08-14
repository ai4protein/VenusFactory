import os
import sys
import time
from pathlib import Path

def check_environment():
    print("Checking running environment...")
    
    if sys.version_info < (3, 8):
        print("Python version too low, requires 3.8 or higher")
        return False
    
    print(f"Python version: {sys.version}")
    
    required_files = [
        "src/webui.py",
        "src/web/stats_api.py",
        "src/web/index_tab.py",
        "src/config/server_config.py"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
                    print(f"Missing required file: {file_path}")
        return False
    
    print("All required files check passed")
    return True

def init_server():
    print("Initializing VenusFactory server...")
    
    try:
        from src.config.server_config import init_server_environment
        if not init_server_environment():
            print("Server environment initialization failed")
            return False
        
        print("Server environment initialization completed")
        return True
        
    except ImportError:
        print("Cannot import server configuration, using default configuration")
        return True
    except Exception as e:
        print(f"Server initialization failed: {e}")
        return False

def start_web_service():
    print("Starting Web service...")
    
    try:
        from src.webui import create_ui
        
        demo = create_ui()
        
        print("Web service started successfully!")
        print("Access address: http://localhost:7860")
        print("Statistics system automatically initialized and running")
        print("Press Ctrl+C to stop service")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\nService stopped")
    except Exception as e:
        print(f"Web service startup failed: {e}")
        return False
    
    return True

def main():
    print("=" * 60)
    print("VenusFactory Server Startup Program")
    print("=" * 60)
    
    if not check_environment():
        print("Environment check failed, cannot start service")
        sys.exit(1)
    
    if not init_server():
        print("Server initialization failed, cannot start service")
        sys.exit(1)
    
    if not start_web_service():
        print("Web service startup failed")
        sys.exit(1)
    
    print("Server startup completed")

if __name__ == "__main__":
    main()
