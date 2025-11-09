"""
Basic usage example
"""
from pipeline import VideoProductionPipeline
from core.config import Config

# Quick start with defaults
pipeline = VideoProductionPipeline()

scenes = ["farm at sunrise", "city marketplace", "walking home"]
narration = "A journey of discovery."

video = pipeline.create_video(
    scene_prompts=scenes,
    narration_text=narration
)

print(f"Video ready: {video}")