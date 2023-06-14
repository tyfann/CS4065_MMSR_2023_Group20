import librosa
from librosa.beat import beat_track


def dpm_detection(music_path):
    y, sr = librosa.load(music_path)

    tempo, beat_frames = beat_track(y=y, sr=sr)
    print(f"Tempo: {tempo:.2f} BPM")

    return tempo


music_path = 'songs/unlock-me-149058.mp3'
dpm_detection(music_path)
