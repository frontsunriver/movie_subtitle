"""Build a Windows folder containing GenerateCaption.exe.

Run from the project root with the virtual environment Python:

    env\\Scripts\\python.exe build_exe.py

The finished program is dist\\GenerateCaption\\GenerateCaption.exe.
Keep the ffmpeg folder next to that exe, or install ffmpeg on PATH.
"""

import PyInstaller.__main__


PyInstaller.__main__.run([
    "main.py",
    "--noconfirm",
    "--windowed",
    "--name", "GenerateCaption",
    "--collect-all", "faster_whisper",
    "--collect-all", "ctranslate2",
    "--collect-all", "onnxruntime",
    "--collect-all", "tokenizers",
    "--collect-all", "transformers",
    "--collect-all", "torch",
    "--collect-submodules", "av",
])
