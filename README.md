# Automation: Faceless Channel Creator

A powerful, open-source pipeline to automate the creation of faceless videos for platforms like YouTube, TikTok, and Instagram. This project leverages cutting-edge AI models to generate content from a single idea to a fully produced video, all without showing your face.

## 🚀 What is this?

This repository provides a set of scripts and tools that automate the workflow for a "faceless" content channel. Instead of recording yourself, you use AI to generate the script, voice, and visuals, then compile them into a engaging video. It's perfect for storytelling, educational content, reddit stories, history explainers, and more.

**Core Philosophy:** Democratize content creation by using free and open-source AI models, giving you full control and ownership over your content without relying on expensive APIs.

## ✨ Features

*   **🧠 AI Script Generation:** Create compelling scripts from a prompt using Large Language Models (LLMs).
*   **🗣️ Text-to-Speech (TTS):** Convert generated scripts into natural, human-like voiceovers.
*   **🎨 Visual Generation:** Create stunning and relevant visuals for your video using AI image generation models.
*   **📹 Video Assembly:** Automatically combine generated images, audio, and background music into a final, polished video file.
*   **⚙️ Modular & Configurable:** Each step of the pipeline is separate, allowing you to swap out models or customize the workflow to your needs.
*   **🔓 100% Open Source:** Built with freely available models, avoiding vendor lock-in and monthly fees.

## 🛠️ Tech Stack & Open Source Models

This project is built with a powerful combination of tools and models:

*   **Script Generation:** Leverages models like **Llama 3**, **Mistral**, or **Phi-3** via `ollama`, `oobabooga's text-generation-webui`, or similar local inference servers.
*   **Text-to-Speech:** Uses high-quality local TTS systems like **XTTS-v2** (Coqui AI) for multi-voice capabilities or **piper** for fast, lightweight synthesis.
*   **Image Generation:** Integrates with **Stable Diffusion** (via `AUTOMATIC1111` WebUI or `ComfyUI`) for creating custom visuals from scene descriptions.
*   **Video Editing:** Utilizes `moviepy` or `ffmpeg` for the final assembly and rendering of the video.
*   **Orchestration:** Python scripts glue everything together.

## 📦 Installation

### Prerequisites

1.  **Python 3.10+**
2.  **FFmpeg** (must be added to your system PATH)
3.  **A capable GPU (Recommended):** Significantly speeds up AI model inference for images and audio.
4.  **AI Model Runtimes:**
    *   **For LLMs:** Install [Ollama](https://ollama.ai/) or [text-generation-webui](https://github.com/oobabooga/text-generation-webui).
    *   **For Images:** Install [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) or [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Spymik/Automation.git
    cd Automation
    ```

2.  **Create a Python virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure the models:**
    *   Follow the installation guides for Ollama/Stable Diffusion etc. on their respective repositories.
    *   Download the desired TTS models (e.g., XTTS-v2) as per their documentation.

## 🏃‍♂️ Quick Start

1.  **Configure your project:** Copy `config.example.yaml` to `config.yaml` and edit the file with your paths and model settings.

    ```yaml
    # config.yaml example
    project_name: "My_Faceless_Video"
    script:
      model: "ollama:llama3"
      topic: "The surprising history of the modern sandwich"
    tts:
      model_path: "./models/xtts_v2"
      speaker_wav: "./voices/my_voice.wav" # Reference audio for voice cloning
    image_generation:
      sd_webui_url: "http://127.0.0.1:7860"
    ```

2.  **Run the pipeline:**
    ```bash
    python main.py --config config.yaml
    ```

3.  **Find your video:** The final rendered video will be in the `output/` directory.

## 📁 Project Structure

Automation/
├── src/
│   ├── script_generator.py   # Handles LLM interaction for script writing
│   ├── tts_engine.py         # Manages text-to-speech conversion
│   ├── image_generator.py    # Creates images from scene descriptions
│   ├── video_editor.py       # Assembles final video
│   └── utils.py              # Helper functions
├── output/                   # Generated videos and assets
├── assets/
│   ├── music/               # Background music files
│   └── voices/              # Reference voice samples for TTS
├── config.yaml              # Main configuration file
├── requirements.txt         # Python dependencies
└── README.md               # This file


## 🔧 Configuration

Detailed configuration options are available in `config.example.yaml`. The main sections include:

- **Script Generation:** Model selection, prompt templates, temperature
- **TTS:** Voice model paths, voice cloning settings, output format
- **Image Generation:** Stable Diffusion settings, style prompts, negative prompts
- **Video:** Resolution, framerate, transition styles, background music

## 🎯 Usage Examples

Create different types of content by modifying the script prompts:

- **Mystery Stories:** "Write a 3-minute script about an unsolved historical mystery"
- **Educational Content:** "Create a script explaining quantum computing for beginners"
- **Fun Facts:** "Generate a script about 5 surprising animal behaviors"
- **Philosophical Topics:** "Write about the concept of time across different cultures"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Here are some areas where you can help:

1.  Add support for new AI models
2.  Improve the video editing capabilities
3.  Create better prompt templates
4.  Optimize performance and add caching
5.  Add support for more video platforms and formats

## 🙏 Acknowledgments

- All the open-source AI model developers and communities
- The teams behind Ollama, Stable Diffusion, XTTS, and other amazing tools
- Contributors and testers who help improve this project

---

**Star this repo if you find it useful! ⭐**

