class SubtitleGenerator:
    def __init__(self):
        print("📝 Subtitle generator ready")

    def transcribe(self, audio_or_video_path, output_dir="output"):
        print("🎧 Transcribing audio with Whisper...")
        try:
            cmd = [
                "whisper", audio_or_video_path,
                "--language", "en",
                "--task", "transcribe",
                "--output_dir", output_dir,
                "--output_format", "srt",
                #"--verbose", "False"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)

            base_name = os.path.splitext(os.path.basename(audio_or_video_path))[0]
            srt_path = os.path.join(output_dir, f"{base_name}.srt")

            if os.path.exists(srt_path):
                print(f"✅ Subtitles generated: {srt_path}")
                return srt_path
            else:
                print(f"⚠️ SRT file not found at {srt_path}")
                return None

        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return None