"""
Setup script for Google Colab
"""
import subprocess
import sys

def install_dependencies():
    """Install all dependencies"""
    print("🔧 Installing dependencies...")
    
    packages = [
        'diffusers>=0.30.0',
        'transformers',
        'accelerate',
        'torch',
        'moviepy',
        'pydub',
        'pysrt',
        'opencv-python',
        'Pillow',
        'numpy',
        'scipy',
        'gTTS',
        'pyyaml'
    ]
    
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install'] + packages + ['--quiet'],
        check=True
    )
    
    print("✅ Dependencies installed!")

def setup_environment():
    """Setup environment"""
    from core.utils import setup_directories
    
    print("📁 Creating directories...")
    setup_directories()
    
    print("✅ Environment ready!")

if __name__ == "__main__":
    install_dependencies()
    setup_environment()