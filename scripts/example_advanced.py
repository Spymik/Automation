"""
Advanced usage with custom config
"""
from pipeline import VideoProductionPipeline
from core.config import Config

# Load high quality preset
config = Config('configs/high_quality.yaml')

# Customize
config.model.video_model = 'wan-animate'
config.video.num_frames = 49

# Initialize
pipeline = VideoProductionPipeline(config)

# Full feature usage
video = pipeline.create_video(
    scene_prompts=scenes,
    narration_text=narration,
    background_music="assets/music/emotional.mp3",
    motion_prompts=motion_prompts,
    require_approval=True
)