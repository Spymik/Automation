# ========================================
# WAN 2.1/2.2 VIDEO GENERATION INTEGRATION
# For animated effects and lip-sync
# ========================================

import subprocess
import sys
import os
import torch
import gc

class WanVideoGenerator:
    def __init__(self, version='2.2', model_path=None):
        """
        Initialize Wan video generation

        Args:
            version: '2.1' or '2.2'
            model_path: Optional custom model path
        """
        self.version = version
        self.model_path = model_path
        self.pipe = None

        print(f"🎬 Initializing Wan {version} Video Generator...")
        self._setup_environment()
        self._load_model()

    def _setup_environment(self):
        """Setup environment and install dependencies"""
        print("   Installing dependencies...")

        try:
            # Install required packages
            packages = [
                "torch",
                "torchvision",
                "diffusers",
                "transformers",
                "accelerate",
                "imageio",
                "imageio-ffmpeg",
                "opencv-python",
                "einops",
                "omegaconf"
            ]

            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + packages + ["--quiet"],
                check=True
            )

            print("✅ Dependencies installed")

        except Exception as e:
            print(f"⚠️ Dependency installation failed: {e}")

    def _load_model(self):
        """Load Wan model"""
        try:
            print(f"   Loading Wan {self.version} model...")

            if self.version == '2.2':
                # Wan 2.2 - Latest version with better quality
                model_id = "Kwai-Kolors/Kolors-Wan2.2"
            else:
                # Wan 2.1 - Stable version
                model_id = "Kwai-Kolors/Kolors-Wan2.1"

            # Use custom path if provided
            if self.model_path:
                model_id = self.model_path

            # Check diffusers version and use appropriate import
            try:
                import diffusers
                from packaging import version
                diffusers_version = version.parse(diffusers.__version__)

                # Try different import strategies based on version
                if diffusers_version >= version.parse("0.29.0"):
                    # Newer versions might have different API
                    from diffusers import CogVideoXImageToVideoPipeline as ImageToVideoPipeline
                else:
                    # For older versions, use standard pipeline
                    from diffusers import DiffusionPipeline as ImageToVideoPipeline

                print(f"   Using diffusers version: {diffusers.__version__}")

            except Exception as e:
                print(f"   Import attempt 1 failed: {e}")
                # Fallback to basic DiffusionPipeline
                from diffusers import DiffusionPipeline as ImageToVideoPipeline

            # Load the pipeline
            try:
                self.pipe = ImageToVideoPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
            except Exception as e:
                print(f"   Trying alternative loading method...")
                # Alternative: Load without variant
                self.pipe = ImageToVideoPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16
                )

            # Move to GPU
            self.pipe.to("cuda")

            # Enable optimizations
            try:
                self.pipe.enable_model_cpu_offload()
                print("   CPU offload enabled")
            except:
                print("   CPU offload not available")

            try:
                self.pipe.enable_vae_slicing()
                print("   VAE slicing enabled")
            except:
                print("   VAE slicing not available")

            print(f"✅ Wan {self.version} loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load Wan model: {e}")
            print("\n" + "="*60)
            print("⚠️ WAN MODEL NOT AVAILABLE")
            print("="*60)
            print("\nWan 2.1/2.2 requires:")
            print("1. Latest diffusers version:")
            print("   !pip install diffusers>=0.29.0 --upgrade")
            print("\n2. Or install from source:")
            print("   !pip install git+https://github.com/huggingface/diffusers.git")
            print("\n3. Alternative: Use manual setup")
            print("   See: https://github.com/Kwai-Kolors/Kolors-Wan")
            print("\n4. OR: Use simpler video methods (effects/cogvideo/svd)")
            print("="*60 + "\n")

            # Set pipe to None so we can handle gracefully
            self.pipe = None
            raise RuntimeError("Wan model loading failed. See instructions above.")

    def generate_video(self, image_path, output_path,
                      motion_prompt=None,
                      num_frames=16,
                      fps=8,
                      motion_scale=1.0,
                      seed=42):
        """
        Generate video from image using Wan

        Args:
            image_path: Path to input image
            output_path: Path for output video
            motion_prompt: Optional text describing motion
            num_frames: Number of frames to generate (16-49)
            fps: Frames per second
            motion_scale: Motion intensity (0.5-2.0)
            seed: Random seed for reproducibility
        """

        if not self.pipe:
            print("❌ Model not loaded")
            return None

        try:
            from PIL import Image
            from diffusers.utils import export_to_video

            print(f"🎬 Generating video from {image_path}...")
            print(f"   Frames: {num_frames}, FPS: {fps}, Motion: {motion_scale}")

            # Load image
            image = Image.open(image_path).convert('RGB')

            # Resize to optimal size (Wan works best with specific resolutions)
            # For 9:16 aspect ratio
            target_size = (576, 1024)  # Width x Height
            image = image.resize(target_size, Image.Resampling.LANCZOS)

            # Generate video frames
            generator = torch.Generator(device="cuda").manual_seed(seed)

            if motion_prompt:
                # With motion prompt (better control)
                print(f"   Motion prompt: {motion_prompt}")
                frames = self.pipe(
                    image=image,
                    prompt=motion_prompt,
                    num_frames=num_frames,
                    guidance_scale=motion_scale,
                    generator=generator,
                    num_inference_steps=25
                ).frames[0]
            else:
                # Without motion prompt (auto-animate)
                frames = self.pipe(
                    image=image,
                    num_frames=num_frames,
                    guidance_scale=motion_scale,
                    generator=generator,
                    num_inference_steps=25
                ).frames[0]

            # Export to video
            export_to_video(frames, output_path, fps=fps)

            print(f"✅ Video generated: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_with_lip_sync(self, image_path, audio_path, output_path,
                               face_region=None):
        """
        Generate video with lip-sync (requires additional tools)

        Args:
            image_path: Path to face image
            audio_path: Path to audio file
            output_path: Output video path
            face_region: Optional (x, y, w, h) for face location
        """

        print("🗣️ Generating lip-sync video...")
        print("⚠️ Note: Wan doesn't directly support lip-sync.")
        print("   For lip-sync, consider using:")
        print("   - Wav2Lip: https://github.com/Rudrabha/Wav2Lip")
        print("   - SadTalker: https://github.com/OpenTalker/SadTalker")
        print("   - Video-Retalking: https://github.com/OpenTalker/video-retalking")

        # For now, just generate animated video
        return self.generate_video(image_path, output_path)

    def batch_generate(self, image_paths, output_dir, motion_prompts=None, **kwargs):
        """
        Generate videos for multiple images

        Args:
            image_paths: List of image paths
            output_dir: Directory for output videos
            motion_prompts: Optional list of motion descriptions
            **kwargs: Additional arguments for generate_video
        """

        os.makedirs(output_dir, exist_ok=True)

        if motion_prompts is None:
            motion_prompts = [None] * len(image_paths)

        video_paths = []

        for i, (img_path, motion) in enumerate(zip(image_paths, motion_prompts), 1):
            print(f"\n[{i}/{len(image_paths)}] Processing {img_path}...")

            output_path = os.path.join(output_dir, f"video_{i:03d}.mp4")

            result = self.generate_video(
                image_path=img_path,
                output_path=output_path,
                motion_prompt=motion,
                **kwargs
            )

            if result:
                video_paths.append(result)

            # Clear GPU cache between generations
            torch.cuda.empty_cache()
            gc.collect()

        print(f"\n✅ Generated {len(video_paths)} videos")
        return video_paths

    def cleanup(self):
        """Free up GPU memory"""
        if self.pipe:
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Wan model unloaded")


# ========================================
# LIP-SYNC INTEGRATION (Wav2Lip)
# ========================================

class LipSyncGenerator:
    def __init__(self, model_type='wav2lip'):
        """
        Initialize lip-sync generator

        Args:
            model_type: 'wav2lip' or 'sadtalker'
        """
        self.model_type = model_type
        print(f"🗣️ Initializing {model_type} for lip-sync...")
        self._setup_lipsync()

    def _setup_lipsync(self):
        """Setup lip-sync tools"""
        try:
            if self.model_type == 'wav2lip':
                self._setup_wav2lip()
            elif self.model_type == 'sadtalker':
                self._setup_sadtalker()
        except Exception as e:
            print(f"⚠️ Lip-sync setup failed: {e}")
            print("   Lip-sync will not be available")

    def _setup_wav2lip(self):
        """Setup Wav2Lip"""
        print("   Setting up Wav2Lip...")

        # Check if already cloned
        if not os.path.exists("/content/Wav2Lip"):
            print("   Cloning Wav2Lip repository...")
            subprocess.run([
                "git", "clone",
                "https://github.com/Rudrabha/Wav2Lip.git",
                "/content/Wav2Lip"
            ], check=True)

        # Install dependencies
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "librosa", "opencv-python", "torch", "torchvision",
            "numpy", "scipy", "tqdm", "--quiet"
        ], check=True)

        # Download pretrained model
        model_path = "/content/Wav2Lip/checkpoints/wav2lip_gan.pth"
        if not os.path.exists(model_path):
            print("   Downloading Wav2Lip model...")
            os.makedirs("/content/Wav2Lip/checkpoints", exist_ok=True)
            subprocess.run([
                "wget",
                "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA",
                "-O", model_path
            ])

        print("✅ Wav2Lip ready")

    def _setup_sadtalker(self):
        """Setup SadTalker"""
        print("   Setting up SadTalker...")

        if not os.path.exists("/content/SadTalker"):
            subprocess.run([
                "git", "clone",
                "https://github.com/OpenTalker/SadTalker.git",
                "/content/SadTalker"
            ], check=True)

        print("✅ SadTalker cloned")

    def generate_lipsync(self, face_image, audio_path, output_path):
        """
        Generate lip-synced video

        Args:
            face_image: Path to face image
            audio_path: Path to audio file
            output_path: Output video path
        """

        if self.model_type == 'wav2lip':
            return self._generate_wav2lip(face_image, audio_path, output_path)
        elif self.model_type == 'sadtalker':
            return self._generate_sadtalker(face_image, audio_path, output_path)

    def _generate_wav2lip(self, face_image, audio_path, output_path):
        """Generate with Wav2Lip"""
        try:
            cmd = [
                "python", "/content/Wav2Lip/inference.py",
                "--checkpoint_path", "/content/Wav2Lip/checkpoints/wav2lip_gan.pth",
                "--face", face_image,
                "--audio", audio_path,
                "--outfile", output_path
            ]

            subprocess.run(cmd, check=True)
            print(f"✅ Lip-sync video generated: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Wav2Lip generation failed: {e}")
            return None

    def _generate_sadtalker(self, face_image, audio_path, output_path):
        """Generate with SadTalker"""
        try:
            cmd = [
                "python", "/content/SadTalker/inference.py",
                "--driven_audio", audio_path,
                "--source_image", face_image,
                "--result_dir", os.path.dirname(output_path)
            ]

            subprocess.run(cmd, check=True, cwd="/content/SadTalker")
            print(f"✅ SadTalker video generated")
            return output_path

        except Exception as e:
            print(f"❌ SadTalker generation failed: {e}")
            return None


# ========================================
# USAGE EXAMPLES
# ========================================

def example_wan_basic():
    """Example: Basic Wan video generation"""

    wan = WanVideoGenerator(version='2.2')

    wan.generate_video(
        image_path="assets/images/scene_001.png",
        output_path="output/wan_video_001.mp4",
        num_frames=24,
        fps=8,
        motion_scale=1.2
    )

    wan.cleanup()


def example_wan_with_motion_prompt():
    """Example: Wan with motion description"""

    wan = WanVideoGenerator(version='2.2')

    # Define motion for each scene
    motion_prompts = [
        "camera slowly zooms in, character walks forward",
        "character turns head, wind blows gently",
        "character raises hand, dramatic lighting shift"
    ]

    image_paths = [
        "assets/images/scene_001.png",
        "assets/images/scene_002.png",
        "assets/images/scene_003.png"
    ]

    videos = wan.batch_generate(
        image_paths=image_paths,
        output_dir="output/wan_videos",
        motion_prompts=motion_prompts,
        num_frames=24,
        fps=8
    )

    wan.cleanup()


def example_lipsync():
    """Example: Lip-sync generation with Wav2Lip"""

    # Generate lip-synced video
    lipsync = LipSyncGenerator(model_type='wav2lip')

    lipsync.generate_lipsync(
        face_image="assets/images/character_face.png",
        audio_path="assets/audio/narration.wav",
        output_path="output/lipsync_video.mp4"
    )


def example_combined_wan_and_lipsync():
    """Example: Combine Wan animation with lip-sync"""

    print("🎬 Combined Wan + Lip-sync Pipeline")

    # Step 1: Generate animated video with Wan
    wan = WanVideoGenerator(version='2.2')

    wan_output = wan.generate_video(
        image_path="assets/images/character.png",
        output_path="output/wan_animated.mp4",
        motion_prompt="character speaking, slight head movement",
        num_frames=24,
        fps=8
    )

    wan.cleanup()

    # Step 2: Apply lip-sync to the animated video
    lipsync = LipSyncGenerator(model_type='wav2lip')

    final_output = lipsync.generate_lipsync(
        face_image=wan_output,  # Use Wan output as input
        audio_path="assets/audio/narration.wav",
        output_path="output/final_animated_lipsync.mp4"
    )

    print(f"✅ Final video with animation and lip-sync: {final_output}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("WAN 2.1/2.2 VIDEO GENERATION")
    print("="*60)

    print("\nFeatures:")
    print("  🎬 Image-to-video animation with Wan")
    print("  🗣️ Lip-sync with Wav2Lip/SadTalker")
    print("  🎨 Motion control with text prompts")
    print("  📹 Batch processing support")

    print("\nNote: First run will download models (~2-4GB)")

class CogVideoXAlternative:
    """
    CogVideoX as alternative to Wan
    Works with standard diffusers
    """

    def __init__(self):
        """Initialize CogVideoX"""
        print("🎬 Initializing CogVideoX (Wan alternative)...")
        self._load_model()

    def _load_model(self):
        """Load CogVideoX model"""
        try:
            from diffusers import CogVideoXPipeline
            import torch

            # Use CogVideoX-2B (lighter) or CogVideoX-5B (better quality)
            model_id = "THUDM/CogVideoX-2b"

            self.pipe = CogVideoXPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            )

            self.pipe.to("cuda")
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()

            print("✅ CogVideoX loaded as alternative")

        except Exception as e:
            print(f"❌ CogVideoX loading failed: {e}")
            self.pipe = None

    def generate_video(self, image_path, output_path,
                      motion_prompt="natural movement",
                      num_frames=24, fps=8):
        """Generate video with CogVideoX"""
        if not self.pipe:
            print("❌ Model not loaded")
            return None

        try:
            from PIL import Image
            import torch

            print(f"🎬 Generating with CogVideoX...")

            # Load image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((720, 480))  # CogVideoX preferred size

            # Generate
            generator = torch.Generator(device="cuda").manual_seed(42)

            video_frames = self.pipe(
                prompt=motion_prompt,
                image=image,
                num_frames=num_frames,
                guidance_scale=6.0,
                generator=generator,
                num_inference_steps=50
            ).frames[0]

            # Save video
            from moviepy.editor import ImageSequenceClip
            import numpy as np

            clip = ImageSequenceClip(
                [np.array(frame) for frame in video_frames],
                fps=fps
            )
            clip.write_videofile(output_path, codec='libx264', verbose=False, logger=None)
            clip.close()

            print(f"✅ Video generated: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None

class EnhancedEffectsGenerator:
    """
    Enhanced version of Ken Burns effects
    Better than basic, lighter than Wan
    """

    def __init__(self):
        """Initialize enhanced effects"""
        print("🎬 Enhanced effects generator ready")

    def generate_video(self, image_path, output_path,
                      effect_type='zoom_pan',
                      duration=3, fps=24):
        """
        Generate video with enhanced effects

        effect_type options:
        - zoom_pan: Zoom in while panning
        - zoom_rotate: Zoom with rotation
        - parallax: 3D parallax effect
        - morph: Morphing effect
        """
        from PIL import Image
        import numpy as np
        from moviepy import VideoClip

        print(f"🎬 Creating enhanced {effect_type} video...")

        # Load image
        img = Image.open(image_path)
        w, h = img.size

        if effect_type == 'zoom_pan':
            clip = self._create_zoom_pan(img, w, h, duration)
        elif effect_type == 'zoom_rotate':
            clip = self._create_zoom_rotate(img, w, h, duration)
        elif effect_type == 'parallax':
            clip = self._create_parallax(img, w, h, duration)
        else:
            clip = self._create_zoom_pan(img, w, h, duration)

        clip = clip.set_fps(fps)
        clip.write_videofile(output_path, codec='libx264',
                           audio=False, verbose=False, logger=None)
        clip.close()

        print(f"✅ Enhanced video created: {output_path}")
        return output_path

    def _create_zoom_pan(self, img, w, h, duration):
        """Zoom in while panning"""
        zoom_factor = 1.3

        def make_frame(t):
            progress = t / duration

            # Zoom
            scale = 1 + (zoom_factor - 1) * progress
            new_w = int(w * scale)
            new_h = int(h * scale)
            zoomed = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Pan (diagonal movement)
            pan_x = int((new_w - w) * progress * 0.7)
            pan_y = int((new_h - h) * progress * 0.3)

            cropped = zoomed.crop((pan_x, pan_y, pan_x + w, pan_y + h))

            return np.array(cropped)

        from moviepy.editor import VideoClip
        return VideoClip(make_frame, duration=duration)

    def _create_zoom_rotate(self, img, w, h, duration):
        """Zoom with subtle rotation"""
        zoom_factor = 1.25
        max_rotation = 5  # degrees

        def make_frame(t):
            progress = t / duration

            # Calculate transformations
            scale = 1 + (zoom_factor - 1) * progress
            angle = max_rotation * np.sin(progress * np.pi)

            # Apply transformations
            new_w = int(w * scale * 1.2)
            new_h = int(h * scale * 1.2)

            zoomed = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            rotated = zoomed.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

            # Center crop
            left = (rotated.width - w) // 2
            top = (rotated.height - h) // 2
            cropped = rotated.crop((left, top, left + w, top + h))

            return np.array(cropped)

        from moviepy import VideoClip
        return VideoClip(make_frame, duration=duration)

    def _create_parallax(self, img, w, h, duration):
        """3D parallax-like effect"""
        def make_frame(t):
            progress = t / duration

            # Simulate depth by scaling different regions differently
            # This is a simplified parallax effect
            scale_bg = 1.0 + progress * 0.05  # Background moves slower
            scale_fg = 1.0 + progress * 0.15  # Foreground moves faster

            # For simplicity, just do zoom with variable speed
            scale = 1.0 + progress * 0.2
            new_w = int(w * scale)
            new_h = int(h * scale)

            zoomed = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Center crop
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            cropped = zoomed.crop((left, top, left + w, top + h))

            return np.array(cropped)

        from moviepy.editor import VideoClip
        return VideoClip(make_frame, duration=duration)