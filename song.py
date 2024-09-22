import librosa
import numpy as np


class Song:

    def __init__(self, filepath: str, title: str = '', artist: str = '', album: str = '', genre: str = '', year: int = 0):
        self.filepath = filepath
        self.title = title
        self.artist = artist
        self.album = album
        self.genre = genre
        self.year = year

    def __str__(self):
        return f'{self.title} by {self.artist}'
    
    @staticmethod
    def calculate_section_heavy_score(audio_signal: np.ndarray, sample_rate: int, start_slice_index: int, end_slice_index: int):
        section = audio_signal[start_slice_index:end_slice_index]
        rms = np.mean(librosa.feature.rms(y=section))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=section, sr=sample_rate))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=section))
        dynamic_range = np.max(section) - np.min(section)
        tempo, _ = librosa.beat.beat_track(y=section, sr=sample_rate)
        
        # Normalize features (example normalization, adjust as needed)
        rms_norm = rms / np.max(rms)
        spectral_centroid_norm = spectral_centroid / np.max(spectral_centroid)
        zero_crossing_rate_norm = zero_crossing_rate / np.max(zero_crossing_rate)
        dynamic_range_norm = dynamic_range / np.max(dynamic_range)
        tempo_norm = tempo / np.max(tempo)
        
        # Calculate section heavy score (example formula, adjust weights as needed)
        section_score = (rms_norm * 0.3 + spectral_centroid_norm * 0.25 + 
                        zero_crossing_rate_norm * 0.2 + dynamic_range_norm * 0.15 +
                        tempo_norm * 0.1) * 100
        return section_score

    def calculate_heavy_score(self, num_sections=10):
        # Load the audio file
        y, sr = librosa.load(self.filepath)
        section_length = len(y) // num_sections if num_sections > 0 else len(y)
        
        section_scores = []
        for i in range(num_sections):
            start = i * section_length
            end = start + section_length
            section_score = Song.calculate_section_heavy_score(y, sr, start, end)
            section_scores.append(section_score)
        
        # Combine section scores (example: weighted average)
        heavy_score = np.mean(section_scores)
        
        return heavy_score
