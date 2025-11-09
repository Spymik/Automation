"""
Configuration management
"""
from dataclasses import dataclass
from typing import Optional, List
import yaml

@dataclass
class ModelConfig:
    """Model configuration"""
    image_model: str = 'sdxl'
    video_model: str = 'svd'
    tts_engine: str = 'kyutai'
    
@dataclass
class VideoConfig:
    """Video generation settings"""
    aspect_ratio: str = '9:16'
    num_frames: int = 25
    fps: int = 8
    quality: str = 'balanced'  # 'fast', 'balanced', 'high'
    
@dataclass
class AudioConfig:
    """Audio settings"""
    music_volume: float = 0.18
    tts_engine: str = 'kyutai'
    enable_lipsync: bool = False

@dataclass
class CaptionConfig:
    """Caption settings"""
    style: str = 'dynamic'  # 'srt', 'dynamic', 'none'
    fontsize: int = 34
    color: str = 'white'
    stroke_color: str = 'black'
    stroke_width: int = 2
    words_per_caption: int = 4

class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.model = ModelConfig()
        self.video = VideoConfig()
        self.audio = AudioConfig()
        self.caption = CaptionConfig()
        
        if config_path:
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configs
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                setattr(self.model, key, value)
        
        if 'video' in config_dict:
            for key, value in config_dict['video'].items():
                setattr(self.video, key, value)
        
        # ... similar for audio and caption
    
    def save_to_yaml(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'video': self.video.__dict__,
            'audio': self.audio.__dict__,
            'caption': self.caption.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f)