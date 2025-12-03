"""
LLaVA-based model wrapper for medical image classification.
Uses instruction-style prompting to return one of two class labels.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from huggingface_hub import snapshot_download
from transformers import LlavaForConditionalGeneration, LlavaProcessor


class LLaVAModelWrapper:
    """
    LLaVA model wrapper for sequential modality evaluation.
    Performs binary classification for two medical classes by prompting the model.
    """

    def __init__(
        self,
        model_name: str = "liuhaotian/llava-v1.6-mistral-7b",
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.class_names = class_names or ["high_grade", "low_grade"]
        if len(self.class_names) != 2:
            raise ValueError("LLaVAModelWrapper currently supports exactly two classes.")
        self.class_names = [name.strip() for name in self.class_names]

        print(f"\nLoading LLaVA model: {model_name}...")
        cache_dir = None
        try:
            self.processor = LlavaProcessor.from_pretrained(model_name)
        except OSError:
            print("Warning: Could not load processor from hub, downloading repository manually...")
            cache_dir = snapshot_download(repo_id=model_name)
            self.processor = LlavaProcessor.from_pretrained(cache_dir)

        model_source = cache_dir or model_name
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_source,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
        except (ValueError, KeyError):
            if cache_dir is None:
                print("Warning: standard model load failed, downloading repository manually...")
                cache_dir = snapshot_download(repo_id=model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self) -> str:
        first, second = self.class_names
        prompt = (
            "You are a medical imaging expert. "
            "Analyze the provided lung imaging slice and decide if it represents "
            f"a {first} tumor or a {second} tumor. "
            f"Respond with exactly one word: {first} or {second}."
        )
        return prompt

    def _predict_single(self, image) -> Dict:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._build_prompt()},
                    {"type": "image"},
                ],
            }
        ]
        template = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=template,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.2,
            )
        response = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )[0]
        response = response.strip().lower()

        first, second = [c.lower() for c in self.class_names]
        if first in response and second in response:
            # pick the one that appears last to reduce bias
            prediction = int(response.rfind(first) < response.rfind(second))
        elif first in response:
            prediction = 0
        elif second in response:
            prediction = 1
        else:
            prediction = 0 if "high" in response else 1

        probs = {self.class_names[0].lower(): 0.0, self.class_names[1].lower(): 0.0}
        chosen_key = self.class_names[prediction].lower()
        probs[chosen_key] = 1.0

        return {
            "prediction": prediction,
            "confidence": 1.0,
            "probabilities": probs,
        }

    def predict(
        self,
        images: Dict[str, object],
        available_modalities: List[str],
        batch_size: int = 1,
        preprocess: bool = False,
        temperature: float = 1.0,
        use_weighted_ensemble: bool = True,
        try_both_swaps: bool = True,
    ) -> Dict:
        # LLaVA handles a single image at a time. Use the first available modality.
        for mod in available_modalities:
            if mod in images and images[mod] is not None:
                image = images[mod]
                if isinstance(image, list):
                    image = image[0]
                return self._predict_single(image)

        raise ValueError("No valid images found for LLaVA prediction.")

