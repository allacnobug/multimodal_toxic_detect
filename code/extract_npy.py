import os
import numpy as np
import librosa
import subprocess
import shutil
import tempfile

if shutil.which("ffmpeg") is None:
    raise RuntimeError("❌ ffmpeg 未安装或不在 PATH 中，请先安装 ffmpeg")

def extract_audio_from_mp4(mp4_path, npy_path, target_sr=16000):
    print(f"Processing {mp4_path} -> {npy_path}")
    tmp_wav = tempfile.mktemp(suffix=".wav")
    
    command = [
        "ffmpeg", "-y", "-i", mp4_path, "-vn",
        "-ac", "1", "-ar", str(target_sr), "-f", "wav", tmp_wav
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"❌ ffmpeg failed for {mp4_path}")
        print(result.stderr.decode())
        return
    
    try:
        y, sr = librosa.load(tmp_wav, sr=target_sr)
        np.save(npy_path, y)
        print(f"✅ Saved {npy_path}, shape={y.shape}, sr={sr}")
    except Exception as e:
        print(f"❌ Failed to load {tmp_wav}: {e}")
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def batch_extract(root_folder):
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".mp4"):
                mp4_path = os.path.join(subdir, file)
                npy_path = os.path.join(subdir, file.replace(".mp4", ".npy"))
                extract_audio_from_mp4(mp4_path, npy_path)

if __name__ == "__main__":
    batch_extract("/root/multimodel-video/Videos/valid_add")
