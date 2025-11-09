"""
Main production pipeline orchestrator
"""
from typing import List, Optional, Dict
from core.config import Config
from core.utils import ProgressTracker, clear_gpu_memory
from generators import ImageGenerator, ImageToVideoGenerator, TTSGenerator
from processors import PromptEnhancer, VideoAssembler, SubtitleGenerator

class VideoProductionPipeline:
    """Main production pipeline"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize pipeline with configuration"""
        self.config = config or Config()
        
        # Initialize components
        self._init_generators()
        self._init_processors()
    
    def _init_generators(self):
        """Initialize generators"""
        self.image_gen = None  # Lazy loading
        self.video_gen = None
        self.audio_gen = None
    
    def _init_processors(self):
        """Initialize processors"""
        self.prompt_enhancer = None
        self.video_assembler = VideoAssembler()
        self.subtitle_gen = SubtitleGenerator()
    
    def create_video(
        self,
        scene_prompts: List[str],
        narration_text: str,
        output_name: str = "final_video.mp4",
        **kwargs
    ) -> str:
        """
        Main video creation method
        
        Args:
            scene_prompts: List of scene descriptions
            narration_text: Narration text
            output_name: Output filename
            **kwargs: Override config settings
        
        Returns:
            Path to final video
        """
        # Merge kwargs with config
        config = self._merge_config(kwargs)
        
        # Track progress
        tracker = ProgressTracker(total_steps=6)
        
        # Step 1: Generate images
        tracker.update("Generating images...")
        image_paths = self._generate_images(scene_prompts, config)
        clear_gpu_memory()
        
        # Step 2: Generate videos
        tracker.update("Creating video clips...")
        clip_paths = self._generate_videos(image_paths, config)
        clear_gpu_memory()
        
        # Step 3: Merge clips
        tracker.update("Merging clips...")
        merged_video = self._merge_clips(clip_paths)
        
        # Step 4: Generate audio
        tracker.update("Generating audio...")
        audio_path = self._generate_audio(narration_text, config)
        
        # Step 5: Add audio & music
        tracker.update("Adding audio and music...")
        video_with_audio = self._add_audio(merged_video, audio_path, config)
        
        # Step 6: Add captions
        tracker.update("Adding captions...")
        final_video = self._add_captions(video_with_audio, narration_text, config)
        
        tracker.complete()
        return final_video
    
    def _merge_config(self, kwargs: Dict) -> Config:
        """Merge kwargs with config"""
        # Create copy of config
        config = Config()
        config.model = self.config.model
        config.video = self.config.video
        config.audio = self.config.audio
        config.caption = self.config.caption
        
        # Override with kwargs
        # ... implementation
        
        return config
    
    # ... other methods