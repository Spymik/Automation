"""
Utility functions
"""
import os
import torch
import gc
from pathlib import Path

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'assets/images',
        'assets/clips',
        'assets/audio',
        'assets/music',
        'output/videos',
        'configs'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def clear_gpu_memory():
    """Clear GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()

def get_available_vram():
    """Get available VRAM in GB"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0

def verify_file(path: str, min_size: int = 1000) -> bool:
    """Verify file exists and has content"""
    if not os.path.exists(path):
        return False
    
    return os.path.getsize(path) > min_size

class ProgressTracker:
    """Track progress across pipeline stages"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
    
    def update(self, message: str):
        """Update progress"""
        self.current_step += 1
        progress = (self.current_step / self.total_steps) * 100
        print(f"[{progress:.1f}%] {message}")
    
    def complete(self):
        """Mark as complete"""
        print(f"✅ Complete!")