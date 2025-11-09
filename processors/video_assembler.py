# ========================================
# ENHANCED VIDEO ASSEMBLER
# With Background Music & Caption Support
# ========================================
import os
import subprocess
import sys

# Fix ImageMagick configuration for Colab
def setup_imagemagick():
    """Setup ImageMagick for MoviePy TextClip"""
    try:
        # Install ImageMagick
        subprocess.run(["apt-get", "update"], check=True, capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "imagemagick"], check=True, capture_output=True)

        # Configure MoviePy to use ImageMagick
        import moviepy.config as cfg
        cfg.change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

        print("✅ ImageMagick configured for captions")
        return True
    except Exception as e:
        print(f"⚠️ ImageMagick setup failed: {e}")
        return False

# Run setup
setup_imagemagick()

class VideoAssembler:
    def __init__(self):
        print("🎬 Video assembler ready")

    def merge_clips(self, clip_paths, output_path, transition_duration=0.5):
        """Merge multiple video clips with transitions"""
        print("🔗 Merging video clips...")

        clips = []
        for path in clip_paths:
            clip = VideoFileClip(path)
            #clip = clip.fx(FadeIn, transition_duration)
            clip = clip.with_effects([vfx.FadeIn(transition_duration)])
            #clip = clip.fx(FadeOut, transition_duration)
            clip = clip.with_effects([vfx.FadeOut(transition_duration)])
            clips.append(clip)

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(output_path, fps=24, codec='libx264',
                            audio=False, logger=None)

        for clip in clips:
            clip.close()
        final.close()

        print(f"✅ Merged video saved: {output_path}")
        return output_path

    def add_audio_with_music(self, video_path, narration_path, output_path,
                            background_music_path=None, music_volume=0.2):
        """
        Add narration and background music to video

        Args:
            video_path: Path to video file
            narration_path: Path to narration/voice audio
            output_path: Output video path
            background_music_path: Path to background music (optional)
            music_volume: Volume level for background music (0.0 to 1.0)
        """
        print("🎵 Adding audio to video...")

        video = VideoFileClip(video_path)
        narration = AudioFileClip(narration_path)

        print(f"   Video duration: {video.duration}s")
        print(f"   Narration duration: {narration.duration}s")

        # Adjust narration to match video duration
        if narration.duration > video.duration:
            print(f"   Trimming narration to {video.duration}s")
            narration = narration.subclipped(0, video.duration)
        elif narration.duration < video.duration:
            print(f"   Narration is shorter than video")

        # Process background music if provided
        if background_music_path and os.path.exists(background_music_path):
            print(f"   Loading background music: {background_music_path}")
            music = AudioFileClip(background_music_path)

            print(f"   Background music duration: {music.duration}s")

            # Adjust music volume
            #music = music.volumex(music_volume)
            music = music.with_effects([MultiplyVolume(music_volume)])

            # Loop or trim music to match video duration
            if music.duration < video.duration:
                # Calculate how many loops needed
                loops_needed = int(video.duration / music.duration) + 1
                print(f"   Looping music {loops_needed} times to match video duration")

                # Create looped music
                music_clips = [music] * loops_needed
                music = concatenate_videoclips(music_clips)
                music = music.subclipped(0, video.duration)
            else:
                print(f"   Trimming music to {video.duration}s")
                music = music.subclipped(0, video.duration)

            # Fade in/out for music
            #music = music.audio_FadeIn(2.0).audio_FadeOut(2.0)
            music = music.with_effects([AudioFadeIn(2.0), AudioFadeOut(2.0)])

            # Combine narration and music
            print("   Mixing narration with background music...")
            final_audio = CompositeAudioClip([narration, music])
        else:
            print("   No background music provided, using narration only")
            final_audio = narration

        # Set audio to video
        final = video.with_audio(final_audio)

        # Write final video
        final.write_videofile(
            output_path,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            audio_bitrate='192k',

            logger=None
        )

        # Cleanup
        video.close()
        narration.close()
        if background_music_path and os.path.exists(background_music_path):
            music.close()
        final.close()

        print(f"✅ Video with audio and music saved: {output_path}")
        return output_path

    def add_captions_from_srt(self, video_path, srt_path, output_path,
                             font='Arial', fontsize=24, color='white',
                             bg_color='black', position='bottom'):
        """
        Add captions to video from SRT file using PIL (no ImageMagick needed)

        Args:
            video_path: Input video path
            srt_path: Path to SRT subtitle file
            output_path: Output video path
            font: Font name (e.g., 'Arial', 'Arial-Bold')
            fontsize: Font size in pixels
            color: Text color
            bg_color: Background color (or 'transparent')
            position: 'bottom', 'top', or ('center', 'bottom')
        """
        print("📝 Adding captions from SRT file...")

        if not os.path.exists(srt_path):
            print(f"❌ SRT file not found: {srt_path}")
            return video_path

        try:
            import pysrt
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            from moviepy import ImageSequenceClip

            # Load video
            video = VideoFileClip(video_path)

            # Load subtitles
            subs = pysrt.open(srt_path)
            print(f"   Loaded {len(subs)} subtitle entries")

            # Get PIL font
            try:
                # Try to load TrueType font
                pil_font = ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
            except:
                # Fallback to default
                pil_font = ImageFont.load_default()

            # Color mapping
            color_map = {
                'white': (255, 255, 255),
                'black': (0, 0, 0),
                'yellow': (255, 255, 0),
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255)
            }
            text_color = color_map.get(color.lower(), (255, 255, 255))

            # Process video frame by frame
            def make_frame(t):
                # Get original frame
                frame = video.get_frame(t)
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)

                # Find current subtitle
                current_sub = None
                for sub in subs:
                    start_time = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000
                    end_time = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000

                    if start_time <= t < end_time:
                        current_sub = sub
                        break

                # Draw subtitle if exists
                if current_sub:
                    text = current_sub.text.replace('\n', ' ')

                    # Get text size
                    bbox = draw.textbbox((0, 0), text, font=pil_font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Calculate position
                    x = (video.w - text_width) // 2
                    if position == 'bottom':
                        y = video.h - text_height - 50
                    elif position == 'top':
                        y = 50
                    else:
                        y = video.h - text_height - 50

                    # Draw background if needed
                    if bg_color and bg_color != 'transparent':
                        bg_rect = [x - 10, y - 5, x + text_width + 10, y + text_height + 5]
                        draw.rectangle(bg_rect, fill=(0, 0, 0, 180))

                    # Draw text with outline
                    outline_range = 2
                    for ox in range(-outline_range, outline_range + 1):
                        for oy in range(-outline_range, outline_range + 1):
                            if ox != 0 or oy != 0:
                                draw.text((x + ox, y + oy), text, font=pil_font, fill=(0, 0, 0))

                    # Draw main text
                    draw.text((x, y), text, font=pil_font, fill=text_color)

                return np.array(img)

            print("   Rendering video with captions...")
            #final = video.fl(lambda gf, t: make_frame(t))
            final = VideoClip(make_frame, duration=video.duration)



            # Write output
            final.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                logger=None
            )

            # Cleanup
            video.close()
            final.close()

            print(f"✅ Video with captions saved: {output_path}")
            return output_path

        except ImportError:
            print("❌ pysrt not installed. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pysrt", "--quiet"], check=True)
            print("✅ pysrt installed. Please run again.")
            return video_path
        except Exception as e:
            print(f"❌ Error adding captions: {e}")
            import traceback
            traceback.print_exc()
            return video_path

    def add_dynamic_captions(self, video_path, narration_text, output_path,
                           font='Arial-Bold', fontsize=28, color='yellow',
                           stroke_color='black', stroke_width=2,
                           position='bottom', words_per_caption=5):
        """
        Add dynamic word-by-word captions (TikTok/Reels style) using PIL

        Args:
            video_path: Input video path
            narration_text: Full narration text
            output_path: Output video path
            font: Font name (unused, uses DejaVu)
            fontsize: Font size
            color: Text color
            stroke_color: Outline color
            stroke_width: Outline width
            position: Caption position
            words_per_caption: Number of words per caption segment
        """
        print("🎬 Adding dynamic captions...")

        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np

            video = VideoFileClip(video_path)

            # Split text into words
            words = narration_text.split()
            total_words = len(words)

            # Calculate timing (distribute evenly across video)
            time_per_word = video.duration / total_words

            # Get PIL font
            try:
                pil_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
            except:
                pil_font = ImageFont.load_default()

            # Color mapping
            color_map = {
                'white': (255, 255, 255),
                'black': (0, 0, 0),
                'yellow': (255, 255, 0),
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255)
            }
            text_color = color_map.get(color.lower(), (255, 255, 0))
            outline_color = color_map.get(stroke_color.lower(), (0, 0, 0))

            # Create caption segments
            caption_segments = []
            for i in range(0, total_words, words_per_caption):
                segment_words = words[i:i + words_per_caption]
                segment_text = ' '.join(segment_words)

                start_time = i * time_per_word
                end_time = (i + len(segment_words)) * time_per_word

                caption_segments.append({
                    'text': segment_text,
                    'start': start_time,
                    'end': end_time
                })

            print(f"   Created {len(caption_segments)} caption segments")

            # Process video frame by frame
            def make_frame(t):
                # Get original frame
                frame = video.get_frame(t)
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)

                # Find current caption with fade effect
                current_caption = None
                fade_duration = 0.2
                alpha = 1.0

                for seg in caption_segments:
                    if seg['start'] <= t <= seg['end']:
                        current_caption = seg['text']

                        # Fade in
                        if t < seg['start'] + fade_duration:
                            alpha = (t - seg['start']) / fade_duration
                        # Fade out
                        elif t > seg['end'] - fade_duration:
                            alpha = (seg['end'] - t) / fade_duration
                        else:
                            alpha = 1.0

                        alpha = max(0, min(1, alpha))
                        break

                # Draw caption if exists
                if current_caption and alpha > 0:
                    # Get text size
                    bbox = draw.textbbox((0, 0), current_caption, font=pil_font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Calculate position
                    x = (video.w - text_width) // 2
                    if position == 'bottom':
                        y = video.h - text_height - 80
                    elif position == 'top':
                        y = 80
                    else:
                        y = (video.h - text_height) // 2

                    # Draw semi-transparent background box
                    padding = 15
                    bg_alpha = int(128 * alpha)
                    bg_rect = [x - padding, y - padding//2,
                              x + text_width + padding, y + text_height + padding//2]

                    # Create overlay for transparency
                    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.rectangle(bg_rect, fill=(0, 0, 0, bg_alpha))

                    # Composite overlay
                    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                    draw = ImageDraw.Draw(img)

                    # Apply alpha to colors
                    text_color_alpha = tuple([int(c * alpha) for c in text_color])
                    outline_color_alpha = tuple([int(c * alpha) for c in outline_color])

                    # Draw text outline
                    for ox in range(-stroke_width, stroke_width + 1):
                        for oy in range(-stroke_width, stroke_width + 1):
                            if ox != 0 or oy != 0:
                                draw.text((x + ox, y + oy), current_caption,
                                        font=pil_font, fill=outline_color_alpha)

                    # Draw main text
                    draw.text((x, y), current_caption, font=pil_font, fill=text_color_alpha)

                return np.array(img)

            print("   Rendering video with dynamic captions...")
            #final = video.fl(lambda gf, t: make_frame(t))
            final = VideoClip(make_frame, duration=video.duration)

            # Write output
            final.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                logger=None
            )

            # Cleanup
            video.close()
            final.close()

            print(f"✅ Video with dynamic captions saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error adding dynamic captions: {e}")
            import traceback
            traceback.print_exc()
            return video_path