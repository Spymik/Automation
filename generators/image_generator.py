from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import gc

class ImageGenerator:
    def __init__(self, model_type="sdxl", model_id=None,
                 use_character_consistency=False,
                 character_images=None,
                 consistency_method="ipadapter"):
        """
        Initialize image generation pipeline with support for multiple models

        Args:
            model_type: "sdxl", "sd15", "flux", "playground"
            model_id: Custom model ID (optional, uses defaults if None)
            use_character_consistency: Enable character consistency features
            character_images: List of reference character images (paths or PIL Images)
            consistency_method: "ipadapter", "controlnet", or "inpaint"
        """
        self.model_type = model_type
        self.use_character_consistency = use_character_consistency
        self.character_images = character_images
        self.consistency_method = consistency_method

        print(f"🎨 Loading {model_type.upper()} image generation model...")

        # Default model IDs for each type
        default_models = {
            "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
            "sd15": "runwayml/stable-diffusion-v1-5",
            "flux": "black-forest-labs/FLUX.1-schnell",
            "playground": "playgroundai/playground-v2.5-1024px-aesthetic",
            "dreamshaper": "Lykon/DreamShaper",
            "realistic_vision": "SG161222/Realistic_Vision_V6.0_B1_noVAE"
        }

        if model_id is None:
            model_id = default_models.get(model_type, default_models["sdxl"])

        self.model_id = model_id

        # Load appropriate pipeline based on requirements
        if use_character_consistency:
            self._load_with_character_consistency()
        else:
            self._load_standard_pipeline()

        # Enable memory optimizations
        self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()

        # Optional: enable xformers if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ xformers enabled for faster generation")
        except:
            print("⚠️ xformers not available, using default attention")

        print(f"✅ Image generator ready: {model_id}")

    def _load_standard_pipeline(self):
        """Load standard pipeline without character consistency"""
        if self.model_type in ["sdxl", "playground"]:
            from diffusers import StableDiffusionXLPipeline
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
        elif self.model_type == "flux":
            from diffusers import FluxPipeline
            self.pipe = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16
            )
        else:  # SD 1.5 and others
            from diffusers import StableDiffusionPipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                safety_checker=None
            )

        # Optimize scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to("cuda")

    def _load_with_character_consistency(self):
        """Load pipeline with character consistency features"""
        print(f"🎭 Loading with {self.consistency_method} for character consistency...")

        if self.consistency_method == "ipadapter":
            # IP-Adapter for character consistency
            from diffusers import StableDiffusionXLPipeline
            from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLImg2ImgPipeline

            # Try to load IP-Adapter
            try:
                # Install IP-Adapter if needed
                subprocess.run([sys.executable, "-m", "pip", "install",
                              "ip_adapter", "--quiet"], check=False)

                from ip_adapter import IPAdapterXL

                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
                self.pipe.to("cuda")

                # Load IP-Adapter
                self.ip_adapter = IPAdapterXL(
                    self.pipe,
                    "h94/IP-Adapter",
                    "sdxl_models/ip-adapter_sdxl.bin",
                    device="cuda"
                )
                print("✅ IP-Adapter loaded for character consistency")

            except Exception as e:
                print(f"⚠️ IP-Adapter failed to load: {e}")
                print("Falling back to standard pipeline")
                self._load_standard_pipeline()
                self.use_character_consistency = False

        elif self.consistency_method == "controlnet":
            # ControlNet for pose/structure preservation
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

            try:
                controlnet = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-canny-sdxl-1.0",
                    torch_dtype=torch.float16
                )

                self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    self.model_id,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
                self.pipe.to("cuda")
                print("✅ ControlNet loaded for character consistency")

            except Exception as e:
                print(f"⚠️ ControlNet failed to load: {e}")
                print("Falling back to standard pipeline")
                self._load_standard_pipeline()
                self.use_character_consistency = False

        elif self.consistency_method == "inpaint":
            # Inpainting for background replacement
            from diffusers import StableDiffusionXLInpaintPipeline

            try:
                self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16"
                )
                self.pipe.to("cuda")
                print("✅ Inpainting pipeline loaded for background changes")

            except Exception as e:
                print(f"⚠️ Inpaint pipeline failed to load: {e}")
                print("Falling back to standard pipeline")
                self._load_standard_pipeline()
                self.use_character_consistency = False

        else:
            print(f"⚠️ Unknown consistency method: {self.consistency_method}")
            self._load_standard_pipeline()

    def generate_scene(self, prompt, output_path,
                      width=576, height=1024,  # 9:16 aspect ratio for vertical video
                      num_inference_steps=25,
                      guidance_scale=7.5,
                      character_image=None,
                      ip_adapter_scale=0.7):
        """Generate a single scene image"""
        print(f"🖼️ Generating: {prompt[:50]}...")

        # Use character consistency if enabled
        if self.use_character_consistency and character_image:
            return self._generate_with_character(
                prompt, output_path, character_image,
                width, height, num_inference_steps,
                guidance_scale, ip_adapter_scale
            )

        # Standard generation
        if self.model_type == "flux":
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                width=width,
                height=height
            ).images[0]
        else:
            image = self.pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, ugly, bad anatomy",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]

        image.save(output_path)
        print(f"✅ Saved: {output_path}")

        return output_path

    def _generate_with_character(self, prompt, output_path, character_image,
                                width, height, num_inference_steps,
                                guidance_scale, ip_adapter_scale):
        """Generate scene with character consistency"""
        from PIL import Image

        # Load character image if it's a path
        if isinstance(character_image, str):
            character_image = Image.open(character_image).convert('RGB')

        if self.consistency_method == "ipadapter":
            # IP-Adapter generation
            try:
                image = self.ip_adapter.generate(
                    pil_image=character_image,
                    prompt=prompt,
                    negative_prompt="blurry, low quality, distorted",
                    scale=ip_adapter_scale,
                    num_samples=1,
                    num_inference_steps=num_inference_steps,
                    seed=42
                )[0]

            except Exception as e:
                print(f"⚠️ IP-Adapter generation failed: {e}")
                print("Falling back to standard generation")
                return self.generate_scene(prompt, output_path, width, height,
                                          num_inference_steps, guidance_scale, None)

        elif self.consistency_method == "controlnet":
            # ControlNet generation with Canny edge detection
            import cv2
            import numpy as np

            # Convert to numpy and detect edges
            char_np = np.array(character_image)
            edges = cv2.Canny(char_np, 100, 200)
            edges = Image.fromarray(edges)

            image = self.pipe(
                prompt=prompt,
                image=edges,
                negative_prompt="blurry, low quality",
                num_inference_steps=num_inference_steps,
                controlnet_conditioning_scale=0.8,
                width=width,
                height=height
            ).images[0]

        elif self.consistency_method == "inpaint":
            # Inpainting - need a mask (auto-generate or provide)
            # For simplicity, we'll create a simple background mask
            mask = Image.new('L', character_image.size, 255)
            # This is a placeholder - you'd want better masking

            image = self.pipe(
                prompt=prompt,
                image=character_image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height
            ).images[0]

        image.save(output_path)
        print(f"✅ Saved with character: {output_path}")
        return output_path

    def generate_scenes(self, scene_prompts, character_images=None,
                       require_approval=False, max_retries=3):
        """
        Generate multiple scene images with optional character consistency

        Args:
            scene_prompts: List of prompts for each scene
            character_images: Optional character images for consistency
            require_approval: If True, ask for user approval before proceeding
            max_retries: Maximum regeneration attempts per scene
        """
        paths = []

        for i, prompt in enumerate(scene_prompts, 1):
            path = f"assets/images/scene_{i:03d}.png"
            approved = False
            retry_count = 0

            while not approved and retry_count < max_retries:
                # Use character image if provided
                char_img = None
                if character_images:
                    char_img = character_images[i-1] if i-1 < len(character_images) else character_images[0]
                elif self.character_images:
                    char_img = self.character_images[0]

                self.generate_scene(prompt, path, character_image=char_img)

                # Show image and get approval if required
                if require_approval:
                    from IPython.display import display, Image as IPImage, clear_output

                    print(f"\n{'='*60}")
                    print(f"Scene {i}/{len(scene_prompts)}: Review Generated Image")
                    print(f"{'='*60}")
                    print(f"Prompt: {prompt}\n")

                    # Display the image
                    display(IPImage(filename=path))

                    # Get user feedback
                    print("\nOptions:")
                    print("  [a] Approve and continue")
                    print("  [r] Regenerate this image")
                    print("  [e] Edit prompt and regenerate")
                    print("  [s] Skip approval for remaining images")

                    choice = input("\nYour choice (a/r/e/s): ").lower().strip()

                    if choice == 'a':
                        approved = True
                        print("✅ Image approved!")
                    elif choice == 'r':
                        retry_count += 1
                        print(f"🔄 Regenerating... (Attempt {retry_count}/{max_retries})")
                    elif choice == 'e':
                        new_prompt = input(f"Enter new prompt (or press Enter to keep current): ").strip()
                        if new_prompt:
                            prompt = new_prompt
                            scene_prompts[i-1] = new_prompt  # Update for future reference
                        retry_count += 1
                        print(f"🔄 Regenerating with new prompt... (Attempt {retry_count}/{max_retries})")
                    elif choice == 's':
                        approved = True
                        require_approval = False  # Disable approval for rest
                        print("⏭️ Skipping approval for remaining images")
                    else:
                        print("Invalid choice, treating as 'approve'")
                        approved = True

                    clear_output(wait=False)
                else:
                    approved = True

            if not approved:
                print(f"⚠️ Max retries reached for scene {i}. Using last generated image.")

            paths.append(path)

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        return paths

    def cleanup(self):
        """Free up GPU memory"""
        if hasattr(self, 'ip_adapter'):
            del self.ip_adapter
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()