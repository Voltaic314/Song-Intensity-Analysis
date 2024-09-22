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
        self.audio_signal = None
        self.sample_rate = None
        self.get_audio_signal_and_sample_rate()

    def __str__(self):
        return f'{self.title} by {self.artist}'

    def get_audio_signal_and_sample_rate(self):
        self.audio_signal, self.sample_rate = librosa.load(self.filepath)
    
    @property
    def duration(self):
        duration = librosa.get_duration(y=self.audio_signal, sr=self.sample_rate)
        return duration

    @property
    def tempo(self):
        tempo, _ = librosa.beat.beat_track(y=self.audio_signal, sr=self.sample_rate)
        return tempo
    
    @staticmethod
    def intensity_dict(rms: float = None, spectral_centroid: float = None, zero_crossing_rate: float = None, dynamic_range: float = None, tempo: float = None):
        intensity_dict = {
            'rms': rms,
            'spectral_centroid': spectral_centroid,
            'zero_crossing_rate': zero_crossing_rate,
            'dynamic_range': dynamic_range,
            'tempo': tempo
        }
        return intensity_dict

    # Calculate global max values to normalize sections
    def calculate_global_max_values(self):
        rms = np.mean(librosa.feature.rms(y=self.audio_signal))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=self.audio_signal, sr=self.sample_rate))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=self.audio_signal))
        dynamic_range = np.max(self.audio_signal) - np.min(self.audio_signal)
        tempo, _ = librosa.beat.beat_track(y=self.audio_signal, sr=self.sample_rate)
        
        return Song.intensity_dict(rms, spectral_centroid, zero_crossing_rate, dynamic_range, tempo)
    
    @staticmethod
    def calculate_section_intensity(audio_signal: np.ndarray, sample_rate: int, start_slice_index: int, end_slice_index: int, global_max_values: dict):
        section = audio_signal[start_slice_index:end_slice_index]
        rms = np.mean(librosa.feature.rms(y=section))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=section, sr=sample_rate))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=section))
        dynamic_range = np.max(section) - np.min(section)
        tempo, _ = librosa.beat.beat_track(y=section, sr=sample_rate)
        
        # Log-scale normalization (log1p to avoid log(0))
        rms_log = np.log1p(rms)
        spectral_centroid_log = np.log1p(spectral_centroid)
        zero_crossing_rate_log = np.log1p(zero_crossing_rate)
        dynamic_range_log = np.log1p(dynamic_range)
        tempo_log = np.log1p(tempo)
        
        # Normalize features using global max values (log-scaled)
        rms_norm = rms_log / np.log1p(global_max_values['rms']) if global_max_values['rms'] != 0 else 0
        spectral_centroid_norm = spectral_centroid_log / np.log1p(global_max_values['spectral_centroid']) if global_max_values['spectral_centroid'] != 0 else 0
        zero_crossing_rate_norm = zero_crossing_rate_log / np.log1p(global_max_values['zero_crossing_rate']) if global_max_values['zero_crossing_rate'] != 0 else 0
        dynamic_range_norm = dynamic_range_log / np.log1p(global_max_values['dynamic_range']) if global_max_values['dynamic_range'] != 0 else 0
        tempo_norm = tempo_log / np.log1p(global_max_values['tempo']) if global_max_values['tempo'] != 0 else 0
        
        intensity_dict_of_section = Song.intensity_dict(rms_norm, spectral_centroid_norm, zero_crossing_rate_norm, dynamic_range_norm, tempo_norm)

        # Calculate section intensity score (example formula, adjust weights as needed)
        section_score = (rms_norm * 0.3 + spectral_centroid_norm * 0.25 + 
                        zero_crossing_rate_norm * 0.2 + dynamic_range_norm * 0.15 +
                        tempo_norm * 0.1) * 100
        return section_score

    def calculate_intensity(self, num_sections=10):
        # Get global max values for normalization
        global_max_values = self.calculate_global_max_values()
        print(Song.intensity_dict(**global_max_values))
        
        # Load the audio file
        y, sr = librosa.load(self.filepath)
        section_length = len(y) // num_sections if num_sections > 0 else len(y)
        
        section_scores = []
        for i in range(num_sections):
            start = i * section_length
            end = start + section_length
            section_score = Song.calculate_section_intensity(y, sr, start, end, global_max_values)
            section_scores.append(section_score)
        
        # Combine section scores
        intensity = np.mean(section_scores)
        
        return intensity
