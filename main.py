"""
Main entry point
"""
from pipeline import VideoProductionPipeline
from core.config import Config
from core.utils import setup_directories

def main():
    """Main function"""
    
    # Setup
    setup_directories()
    
    # Load configuration
    config = Config('configs/default.yaml')
    
    # Initialize pipeline
    pipeline = VideoProductionPipeline(config)
    
    # Define your story
    scenes = [
        "peaceful farm at sunrise",
        "busy marketplace",
        "person walking home"
    ]
    
    narration = "A story of homecoming."
    
    # Create video
    video = pipeline.create_video(
        scene_prompts=scenes,
        narration_text=narration,
        output_name="my_video.mp4"
    )
    
    print(f"✅ Video created: {video}")

if __name__ == "__main__":
    main()