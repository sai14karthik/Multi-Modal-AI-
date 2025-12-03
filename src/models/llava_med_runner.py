"""
Thin wrapper around the official LLaVA-Med repo to run inference from this project.

We import the builder code from third_party.llava-med and construct the model directly.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch
from PIL import Image

# Add LLaVA-Med repo to path
_llava_med_path = os.path.join(os.path.dirname(__file__), '../../third_party/llava-med')
if os.path.exists(_llava_med_path):
    sys.path.insert(0, _llava_med_path)

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("ACCELERATE_USE_TENSORBOARD", "false")
os.environ.setdefault("USE_TF", "0")

from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


class LLaVAMedRunner:
    """
    Loads LLaVA-Med (e.g., microsoft/llava-med-v1.5-mistral-7b) using the official repo.
    """

    def __init__(
        self,
        model_name: str = "microsoft/llava-med-v1.5-mistral-7b",
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or ["high_grade", "low_grade"]
        self.compute_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        if len(self.class_names) != 2:
            raise ValueError("LLaVAMedRunner currently supports exactly two classes.")

        print(f"\nLoading LLaVA-Med model via official repo: {model_name}...")
        if self.device == "cpu":
            print("⚠️  WARNING: Running on CPU. LLaVA-Med inference will be VERY SLOW (~10-15 min per image).")
            print("   Consider using GPU (--device cuda) or a faster model like CLIP for CPU testing.")
        
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            model_path=model_name,
            model_base=None,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False,  # 4-bit requires GPU support
            device=self.device,
        )
        self.context_len = context_len
        self.model.to(dtype=self.compute_dtype, device=self.device)
        if hasattr(self.model, "model") and hasattr(self.model.model, "mm_projector"):
            self.model.model.mm_projector.to(dtype=self.compute_dtype, device=self.device)
        try:
            vision_tower = self.model.get_vision_tower()
            vision_tower.to(dtype=self.compute_dtype, device=self.device)
        except AttributeError:
            pass
        self.model.eval()
        # Optimize for inference speed
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = True
        self.conv_template = conv_templates["llava_v1"]

    def _build_prompt(self) -> str:
        first, second = self.class_names
        
        # Detect if this is lung cancer grading
        is_grading = any(
            keyword in first.lower() + second.lower()
            for keyword in ["high_grade", "low_grade", "grade"]
        )
        
        if is_grading:
            return (
                "You are a medical imaging expert specializing in lung cancer. "
                "Given this lung CT or PET scan slice, determine whether the lung cancer grade is "
                f"{first} or {second}. Respond with exactly one word: {first} or {second}."
            )
        else:
            return (
                "You are a medical imaging expert. "
                "Given this lung CT or PET scan slice, determine whether it shows "
                f"{first} or {second}. Respond with exactly one word: {first} or {second}."
            )

    @torch.inference_mode()
    def _predict_single(self, image: Image.Image) -> Dict:
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], self._build_prompt())
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Preprocess image once
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(self.device, dtype=self.compute_dtype)
        
        # Tokenize efficiently
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            self.image_processor,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        
        # Set attention mask for efficiency
        attention_mask = torch.ones_like(input_ids)

        # Optimize for speed: reduce tokens, set pad_token, use minimal generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Ultra-fast generation settings for CPU
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=3,  # Minimal tokens - just need one word
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=1,  # Greedy decoding (fastest)
            repetition_penalty=1.0,  # No penalty for speed
        )
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
        first, second = [c.lower() for c in self.class_names]

        if first in output_text and second in output_text:
            prediction = int(output_text.rfind(first) < output_text.rfind(second))
        elif first in output_text:
            prediction = 0
        elif second in output_text:
            prediction = 1
        else:
            prediction = 0 if "high" in output_text else 1

        probs = {first: 0.0, second: 0.0}
        probs[self.class_names[prediction].lower()] = 1.0

        return {
            "prediction": prediction,
            "confidence": 1.0,
            "probabilities": probs,
        }

    @torch.inference_mode()
    def predict(
        self,
        images: Dict[str, Image.Image],
        available_modalities: List[str],
        **_,
    ) -> Dict:
        if not images:
            raise ValueError("LLaVAMedRunner expects at least one image.")
        primary_modality = available_modalities[0] if available_modalities else next(iter(images))
        image = images.get(primary_modality) or next(iter(images.values()))
        return self._predict_single(image)

