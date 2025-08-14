import os
from pathlib import Path

class ServerConfig:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.stats_file = self.base_dir / "src" / "data" / "stats_data.json"
        self.logs_dir = self.base_dir / "logs"
        
        self.stats_config = {
            "auto_save_interval": 30,
            "max_log_size": 100,
            "enable_daily_stats": True,
            "enable_ip_tracking": False,
        }
        
        self.web_config = {
            "host": "0.0.0.0",
            "port": 7860,
            "share": False,
            "debug": False,
            "show_error": True,
        }
        
        self.security_config = {
            "enable_auth": False,
            "max_requests_per_minute": 100,
            "allowed_origins": ["*"],
        }
    
    def create_directories(self):
        try:
            self.logs_dir.mkdir(exist_ok=True)
            print(f"Logs directory created: {self.logs_dir}")
            
            self.stats_file.parent.mkdir(exist_ok=True)
            print(f"Statistics file directory created: {self.stats_file.parent}")
            
        except Exception as e:
            print(f"Failed to create directories: {e}")
    
    def get_stats_file_path(self):
        return str(self.stats_file)
    
    def get_log_file_path(self, log_name="venusfactory"):
        return str(self.logs_dir / f"{log_name}.log")
    
    def validate_config(self):
        errors = []
        
        if not (1024 <= self.web_config["port"] <= 65535):
            errors.append(f"Port number must be between 1024-65535: {self.web_config['port']}")
        
        if self.stats_config["auto_save_interval"] < 10:
            errors.append(f"Auto-save interval cannot be less than 10 seconds: {self.stats_config['auto_save_interval']}")
        
        if errors:
            print("Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("Configuration validation passed")
        return True

server_config = ServerConfig()

def init_server_environment():
    print("Initializing VenusFactory server environment...")
    
    server_config.create_directories()
    
    if not server_config.validate_config():
        print("Server environment initialization failed")
        return False
    
    print("Server environment initialization completed")
    return True

if __name__ == "__main__":
    init_server_environment()
