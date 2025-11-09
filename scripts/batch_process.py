"""
Batch processing multiple videos
"""
from pipeline import VideoProductionPipeline
from core.config import Config
import yaml

# Load batch configuration
with open('batch_config.yaml', 'r') as f:
    batch_config = yaml.safe_load(f)

pipeline = VideoProductionPipeline()

for video_config in batch_config['videos']:
    print(f"\nProcessing: {video_config['name']}")
    
    video = pipeline.create_video(
        scene_prompts=video_config['scenes'],
        narration_text=video_config['narration'],
        output_name=video_config['output_name']
    )
    
    print(f"✅ Complete: {video}")