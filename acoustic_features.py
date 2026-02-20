#!/usr/bin/env python3
"""
Acoustic Features Module for ClinicalWhisper
Extracts acoustic prosody features using OpenSMILE (eGeMAPSv02)
and computes the Zhou Index for anhedonia classification.
"""

import os
import logging
import math
import tempfile
import numpy as np

try:
    import opensmile
    import soundfile as sf
except ImportError:
    opensmile = None
    sf = None

log = logging.getLogger("ClinicalWhisper")

class AcousticExtractor:
    """Extracts eGeMAPSv02 features and computes custom metrics like the Zhou Index."""
    
    def __init__(self):
        self.smile = None
        if opensmile is not None:
            log.info("Loading OpenSMILE eGeMAPSv02 feature set...")
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        else:
            log.warning("opensmile not installed. Acoustic feature extraction will be disabled.")

    def is_available(self):
        return self.smile is not None

    def process_audio_file(self, audio_path: str) -> dict:
        """
        Extract features from a whole audio file.
        Returns a dictionary of aggregated metrics.
        """
        if not self.is_available():
            return {}
            
        try:
            df = self.smile.process_file(audio_path)
            # The returned dataframe has a MultiIndex (file, start, end). We want the first (and only) row as dict.
            return self._extract_metrics(df.iloc[0].to_dict())
        except Exception as e:
            log.error(f"Error processing audio file for acoustics: {e}")
            return {}

    def process_audio_segment(self, audio_data: np.ndarray, sample_rate: int) -> dict:
        """
        Extract features from a numpy array of audio data.
        Writes to a temporary file because opensmile python wrapper prefers files.
        """
        if not self.is_available():
            return {}
            
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            sf.write(tmp_path, audio_data, sample_rate)
            df = self.smile.process_file(tmp_path)
            return self._extract_metrics(df.iloc[0].to_dict())
        except Exception as e:
            log.error(f"Error processing audio segment for acoustics: {e}")
            return {}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _extract_metrics(self, features: dict) -> dict:
        """
        Extract specific parameters from the massive eGeMAPSv02 set 
        and compute the Zhou Index format metrics.
        """
        # F0 variables (Pitch)
        f0_mean = features.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0.0)
        f0_std = features.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0.0) # This is technically the CV in eGeMAPS
        
        # Energy / Loudness variables
        energy_mean = features.get("equivalentSoundLevel_dBp", 0.0)
        energy_std = features.get("loudness_sma3_stddevNorm", 0.0) # CV of loudness
        
        # Jitter and Shimmer
        jitter = features.get("jitterLocal_sma3nz_amean", 0.0)
        shimmer = features.get("shimmerLocaldB_sma3nz_amean", 0.0)
        
        # Speaking rate
        speaking_rate = features.get("equivalentSoundLevel_dBp", 0.0) # fallback
        
        # Compute Zhou Index
        # V_anh = -log(CV_F0 * CV_Energy)
        zhou_index = 0.0
        if f0_std > 0 and energy_std > 0:
            try:
                # eGeMAPS returns stddevNorm which is standard deviation normalized by the mean = Coefficient of Variation (CV)
                cv_f0 = float(f0_std)
                cv_energy = float(energy_std)
                product = cv_f0 * cv_energy
                if product > 0:
                    zhou_index = -math.log(product)
            except Exception as e:
                log.warning(f"Failed to compute Zhou Index: {e}")

        return {
            "pitch_mean_st": round(float(f0_mean), 3),
            "pitch_cv": round(float(f0_std), 3),
            "loudness_mean_db": round(float(energy_mean), 3),
            "loudness_cv": round(float(energy_std), 3),
            "jitter": round(float(jitter), 3),
            "shimmer": round(float(shimmer), 3),
            "zhou_index": round(float(zhou_index), 3)
        }
