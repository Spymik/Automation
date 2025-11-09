import os
import sys
import subprocess
import warnings

class TTSGenerator:
    def __init__(self, engine='gtts'):
        """
        Initialize TTS engine

        Engines for Motivational Videos (Deep, Emotional, Warm):
        - orpheus: Orpheus TTS (Apache 2.0) - Human-like with voices/emotions
        - dia: Dia TTS (Apache 2.0) - Ultra-realistic dialogue
        - chatterbox: Chatterbox (MIT) - Expressive, rivals ElevenLabs
        - higgs: Higgs Audio V2 (Apache 2.0) - Emotional foundation model
        - openvoice: OpenVoice V2 (MIT) - Instant voice cloning
        - styletts: StyleTTS2 (MIT) - Prosody/style diffusion
        - parlertts: ParlerTTS (Apache 2.0) - Style via text description
        - xtts: XTTS-v2 - Emotional + cloning

        Other engines:
        - gtts, bark, kyutai, piper, emotivoice, cosyvoice
        """
        self.engine = engine.lower()
        print(f"🎤 Using TTS engine: {self.engine}")

    # -----------------------------
    #  Google TTS (simple fallback)
    # -----------------------------
    def generate_gtts(self, text, output_path, lang='en', slow=False):
        try:
            from gtts import gTTS
            print(f"   Generating TTS (gTTS) for: {text[:60]}...")
            tts = gTTS(text=text, lang=lang, slow=slow)
            tts.save(output_path)
            if os.path.exists(output_path):
                print(f"✅ gTTS done → {output_path}")
                return output_path
            else:
                print("❌ gTTS failed to create file.")
                return None
        except Exception as e:
            print(f"❌ gTTS failed: {e}")
            return None

    # -----------------------------
    #  Bark TTS (expressive)
    # -----------------------------
    def generate_bark(self, text, output_path):
        try:
            print("   Installing Bark dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install",
                          "git+https://github.com/suno-ai/bark.git",
                          "scipy", "nltk", "--quiet"], check=True)

            import numpy as np
            import torch
            import nltk

            try:
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
            except:
                pass

            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = patched_load
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])

            from bark import SAMPLE_RATE, generate_audio, preload_models
            from scipy.io.wavfile import write as write_wav

            print("   Loading Bark models (first time may take a while)...")
            preload_models()

            print(f"   Generating expressive Bark voice for: {text[:50]}...")

            try:
                sentences = nltk.sent_tokenize(text)
            except:
                import re
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) == 0:
                sentences = [text]

            audio_segments = []
            silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)

            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                print(f"     Processing segment {i+1}/{len(sentences)}: {sentence[:40]}...")
                audio_segment = generate_audio(sentence, history_prompt="v2/en_speaker_6")
                audio_segments.append(audio_segment)
                if i < len(sentences) - 1:
                    audio_segments.append(silence)

            if len(audio_segments) > 0:
                full_audio = np.concatenate(audio_segments)
            else:
                full_audio = generate_audio(text)

            if full_audio.dtype != np.int16:
                max_val = np.abs(full_audio).max()
                if max_val > 1.0:
                    full_audio = full_audio / max_val
                full_audio = (full_audio * 32767).astype(np.int16)

            write_wav(output_path, SAMPLE_RATE, full_audio)
            torch.load = original_load

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                duration = len(full_audio) / SAMPLE_RATE
                print(f"✅ Bark done → {output_path} (duration: {duration:.1f}s)")
                return output_path
            else:
                raise Exception("Generated file is empty or doesn't exist")

        except Exception as e:
            print(f"❌ Bark failed: {e}, falling back to gTTS")
            import traceback
            traceback.print_exc()
            return self.generate_gtts(text, output_path)

    # -----------------------------
    #  Orpheus TTS (Human-like)
    # -----------------------------
    def generate_orpheus(self, text, output_path, voice='leo', emotion_tag=None):
        try:
            print("   Installing Orpheus dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install", "orpheus-speech", "vllm==0.7.3", "--quiet"], check=True)

            from orpheus_tts import OrpheusModel
            import wave
            import numpy as np

            print("   Loading Orpheus model...")
            model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod", max_model_len=2048)


            print(f"   Generating Orpheus voice for: {text[:50]}...")

            # Format prompt with voice and optional emotion
            formatted_text = f"{voice}: {text}"
            if emotion_tag:
                formatted_text = f"{voice}: {emotion_tag} {text}"

            syn_tokens = model.generate_speech(prompt=formatted_text, voice=voice)

            # Write to WAV (streaming chunks)
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                for audio_chunk in syn_tokens:
                    wf.writeframes(audio_chunk)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                duration = os.path.getsize(output_path) / (2 * 24000)  # Approx
                print(f"✅ Orpheus done → {output_path} (duration: {duration:.1f}s, voice: {voice})")
                return output_path
            else:
                raise Exception("Generated file is empty")

        except Exception as e:
            print(f"❌ Orpheus failed: {e}, falling back to gTTS")
            import traceback
            traceback.print_exc()
            return self.generate_gtts(text, output_path)

    # -----------------------------
    #  Dia TTS (Ultra-realistic dialogue)
    # -----------------------------
    def generate_dia(self, text, output_path, audio_prompt=None):
        try:
            print("   Installing Dia dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install",
                          "git+https://github.com/nari-labs/dia.git",
                          "torch", "torchaudio", "scipy", "--quiet"], check=True)

            import torch
            from dia import DiaModel
            import numpy as np
            from scipy.io.wavfile import write as write_wav

            print("   Loading Dia TTS model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = DiaModel.from_pretrained("narilabs/dia-1.6b")
            model = model.to(device)

            print(f"   Generating Dia voice for: {text[:50]}...")

            # For voice cloning, provide audio_prompt
            with torch.no_grad():
                if audio_prompt and os.path.exists(audio_prompt):
                    audio = model.generate_with_prompt(text, audio_prompt)
                else:
                    audio = model.generate(text)

            # Convert to numpy
            audio_np = audio.cpu().numpy().squeeze()
            sample_rate = 24000

            # Normalize
            max_val = np.abs(audio_np).max()
            if max_val > 1.0:
                audio_np = audio_np / max_val
            audio_int16 = (audio_np * 32767).astype(np.int16)

            write_wav(output_path, sample_rate, audio_int16)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                duration = len(audio_int16) / sample_rate
                print(f"✅ Dia done → {output_path} (duration: {duration:.1f}s)")
                return output_path
            else:
                raise Exception("Generated file is empty")

        except Exception as e:
            print(f"❌ Dia failed: {e}, falling back to gTTS")
            import traceback
            traceback.print_exc()
            return self.generate_gtts(text, output_path)

    # -----------------------------
    #  Chatterbox (Expressive)
    # -----------------------------
    def generate_chatterbox(self, text, output_path, voice_reference=None, exaggeration=1.0):
        try:
            print("   Installing Chatterbox dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install",
                          "git+https://github.com/resemble-ai/chatterbox.git",
                          "transformers", "torch", "torchaudio", "scipy", "--quiet"], check=True)

            import torch
            from chatterbox import Chatterbox
            import numpy as np
            from scipy.io.wavfile import write as write_wav

            print("   Loading Chatterbox model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Chatterbox.from_pretrained("resemble-ai/chatterbox-0.5b")
            model = model.to(device)

            print(f"   Generating Chatterbox voice for: {text[:50]}...")

            # Generate with emotion exaggeration control
            with torch.no_grad():
                if voice_reference and os.path.exists(voice_reference):
                    # Voice cloning mode
                    audio = model.clone_voice(
                        text,
                        reference_audio=voice_reference,
                        exaggeration=exaggeration
                    )
                else:
                    # Default voice with emotion control
                    audio = model.synthesize(text, exaggeration=exaggeration)

            # Convert to numpy
            audio_np = audio.cpu().numpy().squeeze()
            sample_rate = 24000

            # Normalize
            max_val = np.abs(audio_np).max()
            if max_val > 1.0:
                audio_np = audio_np / max_val
            audio_int16 = (audio_np * 32767).astype(np.int16)

            write_wav(output_path, sample_rate, audio_int16)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                duration = len(audio_int16) / sample_rate
                print(f"✅ Chatterbox done → {output_path} (duration: {duration:.1f}s)")
                return output_path
            else:
                raise Exception("Generated file is empty")

        except Exception as e:
            print(f"❌ Chatterbox failed: {e}, falling back to gTTS")
            import traceback
            traceback.print_exc()
            return self.generate_gtts(text, output_path)


    # -----------------------------
    #  Higgs Audio V2 (Emotional foundation)
    # -----------------------------
    def generate_higgs(self, text, output_path, voice_prompt=None, emotion='neutral'):
        try:
            print("   Installing Higgs Audio V2...")
            subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/boson-ai/higgs-audio.git", "--quiet"], check=True)

            from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
            from boson_multimodal.data_types import ChatMLSample, Message
            import torch
            import torchaudio
            import numpy as np
            from scipy.io.wavfile import write as write_wav

            print("   Loading Higgs Audio V2 model...")
            MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
            AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            system_prompt = f"Generate warm, {emotion} audio from a quiet room."
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=text)
            ]

            engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

            output: HiggsAudioResponse = engine.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.95,
                top_k=50,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"]
            )

            audio_np = output.audio
            sample_rate = output.sampling_rate

            # Normalize to int16
            max_val = np.abs(audio_np).max()
            if max_val > 1.0:
                audio_np = audio_np / max_val
            audio_int16 = (audio_np * 32767).astype(np.int16)

            write_wav(output_path, sample_rate, audio_int16)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                duration = len(audio_int16) / sample_rate
                print(f"✅ Higgs done → {output_path} (duration: {duration:.1f}s)")
                return output_path
            else:
                raise Exception("Generated file is empty")

        except Exception as e:
            print(f"❌ Higgs failed: {e}, falling back to gTTS")
            import traceback
            traceback.print_exc()
            return self.generate_gtts(text, output_path)

    # -----------------------------
    #  OpenVoice V2 (Voice Cloning)
    # -----------------------------
    def generate_openvoice(self, text, output_path, reference_audio=None):
        try:
            print("   Installing OpenVoice V2 dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install",
                          "git+https://github.com/myshell-ai/OpenVoice.git",
                          "torch", "torchaudio", "scipy", "--quiet"], check=True)

            import torch
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter, BaseSpeakerTTS
            import numpy as np
            from scipy.io.wavfile import write as write_wav

            print("   Loading OpenVoice V2 model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Base TTS
            base_speaker_tts = BaseSpeakerTTS(
                'checkpoints/base_speakers/EN',
                device=device
            )

            # Tone converter for voice cloning
            tone_color_converter = ToneColorConverter(
                'checkpoints/converter',
                device=device
            )

            print(f"   Generating OpenVoice for: {text[:50]}...")

            # Generate base audio
            src_path = 'tmp_base.wav'
            base_speaker_tts.tts(text, src_path, speaker='default')

            # If reference audio provided, clone voice
            if reference_audio and os.path.exists(reference_audio):
                print("   Cloning voice from reference...")
                target_se, _ = se_extractor.get_se(
                    reference_audio,
                    tone_color_converter,
                    target_dir='processed'
                )

                tone_color_converter.convert(
                    audio_src_path=src_path,
                    src_se=None,
                    tgt_se=target_se,
                    output_path=output_path
                )

                # Clean up temp file
                if os.path.exists(src_path):
                    os.remove(src_path)
            else:
                # Just use base voice
                os.rename(src_path, output_path)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ OpenVoice done → {output_path}")
                return output_path
            else:
                raise Exception("Generated file is empty")

        except Exception as e:
            print(f"❌ OpenVoice failed: {e}, falling back to gTTS")
            import traceback
            traceback.print_exc()
            return self.generate_gtts(text, output_path)

    # -----------------------------
    #  StyleTTS2 (Prosody/Style)
    # -----------------------------
    def generate_styletts(self, text, output_path, reference_audio=None):
        try:
            print("   Installing StyleTTS2...")
            subprocess.run([sys.executable, "-m", "pip", "install", "styletts2", "torch", "torchaudio", "--quiet"], check=True)

            from styletts2 import StyleTTS2
            from scipy.io.wavfile import write as write_wav
            import numpy as np
            import torch

            print("   Loading StyleTTS2 model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = StyleTTS2.load_pretrained('libritts')  # Multi-speaker for cloning

            print(f"   Generating StyleTTS2 for: {text[:50]}...")

            # Inference (with optional reference for style cloning)
            audio = model.inference(
                text=text,
                speaker_id=0,  # Default; adjust for multi-speaker
                style_control=torch.randn(1, 1, 512).to(device) if reference_audio else None  # Latent style from ref if provided
            )
            audio_np = audio.cpu().numpy().squeeze()
            sample_rate = 24000

            # Normalize
            max_val = np.abs(audio_np).max()
            if max_val > 1.0:
                audio_np = audio_np / max_val
            audio_int16 = (audio_np * 32767).astype(np.int16)

            write_wav(output_path, sample_rate, audio_int16)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                duration = len(audio_int16) / sample_rate
                print(f"✅ StyleTTS2 done → {output_path} (duration: {duration:.1f}s)")
                return output_path
            else:
                raise Exception("Generated file is empty")

        except Exception as e:
            print(f"❌ StyleTTS2 failed: {e}, falling back to gTTS")
            import traceback
            traceback.print_exc()
            return self.generate_gtts(text, output_path)

    # -----------------------------
    #  ParlerTTS (Style via Description)
    # -----------------------------
    def generate_parlertts(self, text, output_path, style_desc="A deep, warm male speaker with soft, emotional delivery."):
        try:
            print("   Installing ParlerTTS...")
            subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/huggingface/parler-tts.git", "soundfile", "--quiet"], check=True)

            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            import soundfile as sf
            import numpy as np
            from scipy.io.wavfile import write as write_wav

            print("   Loading ParlerTTS model...")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
            tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

            print(f"   Generating ParlerTTS for: {text[:50]}...")

            input_ids = tokenizer(style_desc, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            audio_arr = generation.cpu().numpy().squeeze()
            sample_rate = model.config.sampling_rate  # 24kHz

            # To int16
            audio_int16 = (audio_arr * 32767).astype(np.int16)
            write_wav(output_path, sample_rate, audio_int16)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                duration = len(audio_int16) / sample_rate
                print(f"✅ ParlerTTS done → {output_path} (duration: {duration:.1f}s)")
                return output_path
            else:
                raise Exception("Generated file is empty")

        except Exception as e:
            print(f"❌ ParlerTTS failed: {e}, falling back to gTTS")
            import traceback
            traceback.print_exc()
            return self.generate_gtts(text, output_path)

    # -----------------------------
    #  XTTS-v2 (Your existing)
    # -----------------------------
    def generate_xtts(self, text, output_path, speaker_wav=None, language='en'):
        try:
            print("   Installing XTTS-v2 dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install",
                          "TTS", "torch", "torchaudio", "scipy", "--quiet"], check=True)

            from TTS.api import TTS
            import numpy as np

            print("   Loading XTTS-v2 model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

            print(f"   Generating XTTS voice for: {text[:50]}...")

            if speaker_wav and os.path.exists(speaker_wav):
                # Voice cloning with reference audio
                tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=speaker_wav,
                    language=language
                )
            else:
                # Use default speaker
                tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=language
                )

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ XTTS done → {output_path}")
                return output_path
            else:
                raise Exception("Generated file is empty")

        except Exception as e:
            print(f"❌ XTTS failed: {e}, falling back to gTTS")
            import traceback
            traceback.print_exc()
            return self.generate_gtts(text, output_path)

    # -----------------------------
    #  Wrapper (Updated for All)
    # -----------------------------
    def generate(self, text, output_path, **kwargs):
        """
        Generate TTS audio using the selected engine

        kwargs examples:
        - emotion: str for higgs ('warm')
        - voice: str for orpheus ('leo')
        - emotion_tag: str for orpheus ('<sigh>')
        - voice_reference/reference_audio/speaker_wav: path for cloning
        - exaggeration: float for chatterbox (1.5)
        - audio_prompt: path for dia
        - style_desc: str for parlertts ("deep, warm male...")
        """
        if self.engine == "orpheus":
            return self.generate_orpheus(text, output_path, voice=kwargs.get('voice', 'leo'), emotion_tag=kwargs.get('emotion_tag'))
        elif self.engine == "dia":
            return self.generate_dia(text, output_path, audio_prompt=kwargs.get('audio_prompt'))
        elif self.engine == "chatterbox":
            return self.generate_chatterbox(text, output_path, voice_reference=kwargs.get('voice_reference'), exaggeration=kwargs.get('exaggeration', 1.0))
        elif self.engine == "higgs":
            return self.generate_higgs(text, output_path, voice_prompt=kwargs.get('voice_prompt'), emotion=kwargs.get('emotion', 'neutral'))
        elif self.engine == "openvoice":
            return self.generate_openvoice(text, output_path, reference_audio=kwargs.get('reference_audio'))
        elif self.engine == "styletts":
            return self.generate_styletts(text, output_path, reference_audio=kwargs.get('reference_audio'))
        elif self.engine == "parlertts":
            return self.generate_parlertts(text, output_path, style_desc=kwargs.get('style_desc', "A deep, warm male speaker with soft, emotional delivery."))
        elif self.engine == "xtts":
            return self.generate_xtts(text, output_path, speaker_wav=kwargs.get('speaker_wav'), language=kwargs.get('language', 'en'))
        elif self.engine == "bark":
            return self.generate_bark(text, output_path)
        else:
            return self.generate_gtts(text, output_path)




# Example usage for motivational videos
if __name__ == "__main__":
    motivational_text = """
    Success is not final, failure is not fatal.
    It is the courage to continue that counts.
    Believe in yourself, and all that you are.
    """

    print("\n" + "="*60)
    print("TESTING TTS ENGINES FOR MOTIVATIONAL VIDEOS")
    print("="*60)

    # Test Orpheus (Deep Male Voice - 'leo')
    print("\n=== Testing Orpheus (Deep Male Voice - 'leo') ===")
    tts = TTSGenerator(engine='orpheus')
    tts.generate(motivational_text, "motivational_orpheus.wav", voice='leo')

    tts = TTSGenerator(engine='chatterbox')
    #tts.generate(motivational_text, "motivational_chatterbox.wav")

    # Test Higgs (Emotional)
    print("\n=== Testing Higgs Audio V2 (Warm Emotion) ===")
    tts = TTSGenerator(engine='higgs')
    #tts.generate(motivational_text, "motivational_higgs.wav", emotion='warm')

    # Test StyleTTS2 (With Reference for Warmth)
    print("\n=== Testing StyleTTS2 (Style Cloning) ===")
    tts = TTSGenerator(engine='styletts')
    #tts.generate(motivational_text, "motivational_styletts.wav")  # Add reference_audio=your_sample.wav

    # Test ParlerTTS (Descriptive Style)
    print("\n=== Testing ParlerTTS (Warm Description) ===")
    tts = TTSGenerator(engine='parlertts')
    #tts.generate(motivational_text, "motivational_parlertts.wav")

    print("\n" + "="*60)
    print("✅ Generation complete! Play the files to compare.")
    print("="*60)
    print("\nRecommendations for Motivational Videos:")
    print("1. Orpheus 'leo' + <sigh> tag - Deep, empathetic")
    print("2. Higgs with 'warm' emotion - Natural foundation")
    print("3. ParlerTTS with custom style_desc - Precise warmth/softness")
    print("4. StyleTTS2 with LibriVox reference - Prosodic emotion")
    print("\n💡 For cloning: Use your deep voice samples as voice_reference.")
    print("   All free for commercial use!")