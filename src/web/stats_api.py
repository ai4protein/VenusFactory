import json
import os
import time
from datetime import datetime
from typing import Dict, Any
import threading
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
try:
    from src.config.server_config import server_config
except ImportError:
    # If server configuration cannot be imported, use default configuration
    server_config = None

class StatsManager:
    def __init__(self, stats_file_path: str = None):
        if stats_file_path is None:
            if server_config:
                self.stats_file_path = server_config.get_stats_file_path()
            else:
                self.stats_file_path = "stats_data.json"
        else:
            self.stats_file_path = stats_file_path
            
        self.lock = threading.Lock()
        self.stats_data = self._load_stats()
        self._start_auto_save()
    
    def _load_stats(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.stats_file_path):
                with open(self.stats_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data
        except Exception as e:
            print(f"Failed to load existing statistics data: {e}")
        
        default_stats = {
            "total_visits": 0,
            "agent_usage": 0,
            "mutation_prediction_quick": 0,
            "mutation_prediction_advanced": 0,
            "function_prediction_quick": 0,
            "function_prediction_advanced": 0,
            "daily_visits": {},
            "last_updated": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        try:
            self._save_stats(default_stats)
        except Exception as e:
            print(f"Failed to create statistics data file: {e}")
        
        return default_stats
    
    def _save_stats(self, data=None):
        try:
            with self.lock:
                if data is None:
                    data = self.stats_data
                    data["last_updated"] = datetime.now().isoformat()
                
                os.makedirs(os.path.dirname(self.stats_file_path) if os.path.dirname(self.stats_file_path) else '.', exist_ok=True)
                
                with open(self.stats_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save statistics data: {e}")
    
    def _start_auto_save(self):
        def auto_save():
            while True:
                time.sleep(30)
                self._save_stats()
        
        thread = threading.Thread(target=auto_save, daemon=True)
        thread.start()
    
    def track_visit(self, ip_address: str = None, user_agent: str = None):
        with self.lock:
            self.stats_data["total_visits"] += 1
            
            today = datetime.now().strftime("%Y-%m-%d")
            if today not in self.stats_data["daily_visits"]:
                self.stats_data["daily_visits"][today] = 0
            self.stats_data["daily_visits"][today] += 1
    
    def track_usage(self, module: str):
        with self.lock:
            if module in self.stats_data:
                self.stats_data[module] += 1
            else:
                self.stats_data[module] = 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return self.stats_data.copy()
    
    def reset_stats(self):
        with self.lock:
            self.stats_data = {
                "total_visits": 0,
                "agent_usage": 0,
                "mutation_prediction_quick": 0,
                "mutation_prediction_advanced": 0,
                "function_prediction_quick": 0,
                "function_prediction_advanced": 0,
                "daily_visits": {},
                "last_updated": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat()
            }
            self._save_stats()


stats_manager = StatsManager()

def get_stats():
    return stats_manager.get_stats()

def track_visit(ip_address: str = None, user_agent: str = None):
    stats_manager.track_visit(ip_address, user_agent)
    return {"status": "success", "message": "Visit tracked"}

def track_usage(module: str):
    stats_manager.track_usage(module)
    return {"status": "success", "message": f"Usage of {module} tracked"}

def reset_stats():
    stats_manager.reset_stats()
    return {"status": "success", "message": "Stats reset successfully"}
