"""
VoiceForge-Nextgen - RVC Inferencer
File: app/core/rvc_inferencer.py

Purpose:
    Real RVC inference implementation
    Replaces MOCK inference with actual voice conversion

Dependencies:
    - torch (model inference)
    - numpy
    - app.core.feature_extractor (F0, HuBERT)
    - app.core.pitch_shifter (pitch manipulation)

Data Flow:
    audio → extract_f0 → extract_hubert → model.infer() → output
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
import logging

from app.core.feature_extractor import F0Extractor, HubertExtractor
from app.core.pitch_shifter import PitchShifter

# =========================
# NEW IMPORT (THEO CLAUDE)
# =========================
# NOTE: ADDED – memory profiler integration
from app.core.memory_profiler import get_profiler

logger = logging.getLogger("RVCInferencer")


class RVCInferencer:
    """
    Real-time RVC inference engine

    Pipeline:
        1. Extract F0 (pitch)
        2. Extract HuBERT (content)
        3. Apply pitch shift
        4. Run RVC model
        5. Index retrieval (optional)
    """

    def __init__(
        self, model_data: Dict[str, Any], device: str = "cuda", use_fp16: bool = True
    ):
        """
        Initialize inferencer
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_fp16 = use_fp16 and self.device == "cuda"

        # Extract components
        self.model = model_data["model"].to(self.device)
        self.index = model_data.get("index")
        self.config = model_data.get("config", {})

        # Set to eval mode
        self.model.eval()

        # Get model config
        self.sample_rate = self.config.get("sample_rate", 48000)
        self.f0_method = self.config.get("f0_method", "harvest")

        # Initialize feature extractors
        self.f0_extractor = F0Extractor(method=self.f0_method, sr=self.sample_rate)

        self.hubert_extractor = HubertExtractor(device=self.device)

        # Initialize pitch shifter
        self.pitch_shifter = PitchShifter()

        # =========================
        # NEW: PROFILER (CLAUDE)
        # =========================
        # NOTE: ADDED – memory & time profiling
        self.profiler = get_profiler()

        # =========================
        # NEW: INFERENCE COUNTER
        # =========================
        # NOTE: ADDED – track number of inferences
        self.inference_count = 0

        # =========================
        # NEW: AUTO CACHE CLEAR
        # =========================
        # NOTE: ADDED – CUDA cache hygiene
        self.auto_clear_cache = True
        self.clear_cache_interval = 100

        logger.info(
            f"RVCInferencer initialized: "
            f"device={self.device}, fp16={self.use_fp16}, "
            f"f0_method={self.f0_method}"
        )

    def infer(
        self,
        audio: np.ndarray,
        pitch_shift: int = 0,
        index_rate: float = 0.5,
        filter_radius: int = 3,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ) -> np.ndarray:
        """
        Run inference on audio
        """
        if audio.size == 0:
            return audio

        try:
            # =========================
            # NEW: PROFILE INFERENCE
            # =========================
            # NOTE: ADDED – wrap full inference pipeline
            with self.profiler.profile(f"inference_{self.inference_count}"):

                # Step 1: Extract F0
                f0 = self.f0_extractor.extract(audio)

                # Step 2: Shift pitch
                if pitch_shift != 0:
                    f0 = self.pitch_shifter.shift(
                        f0, semitones=pitch_shift, smooth_window=filter_radius
                    )

                # Step 3: Extract HuBERT features
                hubert_features = self.hubert_extractor.extract(
                    audio, sr=self.sample_rate
                )

                # Step 4: Apply index retrieval (if available)
                if self.index is not None and index_rate > 0:
                    hubert_features = self._apply_index(hubert_features, index_rate)

                # Step 5: Run model inference
                output = self._model_forward(hubert_features, f0, protect=protect)

                # Step 6: Mix RMS envelope
                if rms_mix_rate > 0:
                    output = self._mix_rms(audio, output, rms_mix_rate)

            # =========================
            # NEW: AUTO CLEAR CUDA CACHE
            # =========================
            self.inference_count += 1
            if (
                self.auto_clear_cache
                and self.inference_count % self.clear_cache_interval == 0
            ):
                logger.debug(
                    f"Auto clearing CUDA cache " f"(count={self.inference_count})"
                )
                self.clear_cache()

            return output

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return audio

    # ====================================================
    # BELOW: ALL EXISTING METHODS (UNCHANGED)
    # ====================================================
    def _model_forward(
        self, hubert: np.ndarray, f0: np.ndarray, protect: float = 0.33
    ) -> np.ndarray:
        try:
            hubert_tensor = (
                torch.from_numpy(hubert).float().unsqueeze(0).to(self.device)
            )
            f0_tensor = torch.from_numpy(f0).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        output_tensor = self.model(
                            hubert_tensor, f0_tensor, protect=protect
                        )
                else:
                    output_tensor = self.model(
                        hubert_tensor, f0_tensor, protect=protect
                    )

            output = output_tensor.squeeze(0).cpu().numpy()
            return output.astype(np.float32)

        except Exception as e:
            logger.error(f"Model forward failed: {e}")
            raise

    def _apply_index(self, features: np.ndarray, rate: float) -> np.ndarray:
        if self.index is None or rate <= 0:
            return features

        try:
            import faiss

            _, indices = self.index.search(features, k=1)
            retrieved = self.index.reconstruct_batch(indices.flatten())
            mixed = features * (1 - rate) + retrieved * rate
            return mixed.astype(np.float32)

        except Exception as e:
            logger.warning(f"Index retrieval failed: {e}")
            return features

    def _mix_rms(
        self, source: np.ndarray, target: np.ndarray, rate: float
    ) -> np.ndarray:
        if rate <= 0:
            return target

        try:
            source_rms = np.sqrt(np.mean(source**2))
            target_rms = np.sqrt(np.mean(target**2))

            if target_rms < 1e-6:
                return target

            scale = source_rms / target_rms
            mixed = target * (scale**rate)

            peak = np.abs(mixed).max()
            if peak > 1.0:
                mixed = mixed / peak * 0.95

            return mixed.astype(np.float32)

        except Exception as e:
            logger.warning(f"RMS mixing failed: {e}")
            return target

    def clear_cache(self):
        """Clear CUDA cache"""
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def get_stats(self) -> dict:
        return {
            "device": self.device,
            "use_fp16": self.use_fp16,
            "sample_rate": self.sample_rate,
            "f0_method": self.f0_method,
            "has_index": self.index is not None,
            # NOTE: NEW STATS (SAFE ADD)
            "inference_count": self.inference_count,
        }


class MockRVCInferencer:
    """
    Mock inferencer for testing (fallback when real model unavailable)
    """

    def __init__(self, *args, **kwargs):
        logger.warning("Using MOCK inferencer (no real model)")

    def infer(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        return audio * 0.95

    def clear_cache(self):
        pass

    def get_stats(self) -> dict:
        return {"mode": "MOCK"}
