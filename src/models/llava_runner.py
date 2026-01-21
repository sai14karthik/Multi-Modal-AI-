"""
Unified LLaVA runner that works with both regular LLaVA and LLaVA-Med models.
Uses the official LLaVA repository for regular models and LLaVA-Med repo for medical models.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch
from PIL import Image

# Add LLaVA-Med repo to path (supports both regular LLaVA and LLaVA-Med models)
_llava_med_path = os.path.join(os.path.dirname(__file__), '../../third_party/llava-med')

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("ACCELERATE_USE_TENSORBOARD", "false")
os.environ.setdefault("USE_TF", "0")

# Use LLaVA-Med repo (which works and supports Mistral-based LLaVA models)
# The LLaVA-Med builder can handle regular LLaVA Mistral models too
sys.path.insert(0, _llava_med_path)
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


class LLaVARunner:
    """
    Unified LLaVA runner that works with:
    - Regular LLaVA models (e.g., liuhaotian/llava-v1.6-mistral-7b, liuhaotian/llava-v1.6-vicuna-7b)
    - LLaVA-Med models (e.g., microsoft/llava-med-v1.5-mistral-7b)
    """

    def __init__(
        self,
        model_name: str = "liuhaotian/llava-v1.6-mistral-7b",
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        hf_token: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or ["Class0", "Class1"]
        self.compute_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        if len(self.class_names) != 2:
            raise ValueError("LLaVARunner currently supports exactly two classes.")
        
        print(f"\n{'='*60}")
        print(f"Loading LLaVA model: {model_name}")
        print(f"Device: {self.device}")
        print(f"{'='*60}")
        
        if self.device == "cpu":
            print("WARNING: Running on CPU. LLaVA inference will be VERY SLOW (~10-15 min per image).")
            print("   Consider using GPU (--device cuda) or a faster model like CLIP for CPU testing.")
        
        # Get token from parameter or environment variable
        self.hf_token = hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if self.hf_token:
            os.environ['HF_TOKEN'] = self.hf_token
            os.environ['HUGGING_FACE_HUB_TOKEN'] = self.hf_token
        
        # Load model using the builder
        # Note: LLaVA-Med builder supports Mistral-based LLaVA models
        # For Vicuna/Llama-based models, you may need the official LLaVA repo
        try:
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
                model_path=model_name,
                model_base=None,
                model_name=model_name,
                load_8bit=False,
                load_4bit=False,  # 4-bit requires GPU support
                device=self.device,
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nNote: This builder supports Mistral-based LLaVA models.")
            print("For Vicuna/Llama-based models, consider using the official LLaVA repository.")
            raise
        
        self.context_len = context_len
        
        # Move model to device with correct dtype
        self.model.to(dtype=self.compute_dtype, device=self.device)
        
        # Move vision tower and projector if they exist
        if hasattr(self.model, "model") and hasattr(self.model.model, "mm_projector"):
            self.model.model.mm_projector.to(dtype=self.compute_dtype, device=self.device)
        
        try:
            vision_tower = self.model.get_vision_tower()
            if vision_tower is not None:
                vision_tower.to(dtype=self.compute_dtype, device=self.device)
        except AttributeError:
            pass
        
        self.model.eval()
        
        # Optimize for inference
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = True
        
        # Use appropriate conversation template
        if "llava_v1" in conv_templates:
            self.conv_template = conv_templates["llava_v1"]
        elif "llava" in conv_templates:
            self.conv_template = conv_templates["llava"]
        else:
            # Fallback to first available template
            self.conv_template = list(conv_templates.values())[0]
        
        print("Model loaded successfully!\n")

    def _build_prompt(self, previous_predictions: Optional[Dict[str, Dict]] = None, current_modality: Optional[str] = None) -> str:
        first, second = self.class_names
        
        # Build context string if previous predictions are available
        context_parts = []
        if previous_predictions:
            for mod, pred_info in previous_predictions.items():
                pred_class = pred_info.get('class_name', self.class_names[pred_info.get('prediction', 0)])
                context_parts.append(f"the {mod} scan showed {pred_class}")
        
        context_prefix = ""
        if context_parts:
            context_str = ", and ".join(context_parts)
            context_prefix = f"Given that {context_str}, "
        
        # Determine current modality name for prompts (default to generic if not provided)
        if current_modality is None:
            modality_description = "medical imaging scan"
        else:
            modality_description = f"{current_modality} scan"
        
        # Detect if this is cancer grading (generic for any cancer type)
        is_grading = any(
            keyword in first.lower() + second.lower()
            for keyword in ["high_grade", "low_grade", "grade"]
        )
        
        if is_grading:
            return (
                f"{context_prefix}You are a medical imaging expert specializing in cancer diagnosis. "
                f"Given this {modality_description} slice, determine whether the cancer grade is "
                f"{first} or {second}. "
                f"Respond with exactly one word: {first} or {second}."
            )
        else:
            return (
                f"{context_prefix}You are a medical imaging expert. "
                f"Given this {modality_description} slice, determine whether it shows "
                f"{first} or {second}. "
                f"Respond with exactly one word: {first} or {second}."
            )

    @torch.inference_mode()
    def _predict_single(self, image: Image.Image, previous_predictions: Optional[Dict[str, Dict]] = None, current_modality: Optional[str] = None) -> Dict:
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], self._build_prompt(previous_predictions=previous_predictions, current_modality=current_modality))
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Preprocess image
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(
            self.device, dtype=self.compute_dtype
        )
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            self.image_processor,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        
        attention_mask = torch.ones_like(input_ids)

        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Generate prediction (optimized for speed)
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=5,  # Enough for one word response
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=1,  # Greedy decoding (fastest)
            temperature=0.0,  # Deterministic
            repetition_penalty=1.0,
        )
        
        # Decode response
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
        
        # Parse prediction
        first, second = [c.lower() for c in self.class_names]
        
        if first in output_text and second in output_text:
            # Pick the one that appears last (more likely to be the answer)
            prediction = int(output_text.rfind(first) < output_text.rfind(second))
        elif first in output_text:
            prediction = 0
        elif second in output_text:
            prediction = 1
        else:
            # Fallback: default to first class if no match found
            prediction = 0

        # Ensure prediction is valid (0 or 1)
        prediction = max(0, min(1, int(prediction)))

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
        batch_size: int = 1,
        preprocess: bool = False,
        temperature: float = 1.0,
        use_weighted_ensemble: bool = True,
        try_both_swaps: bool = True,
        previous_predictions: Optional[Dict[str, Dict]] = None,
        **_,
    ) -> Dict:
        """
        Predict using LLaVA model.
        For multimodal inputs, uses the first available modality.
        
        Args:
            previous_predictions: Optional dict mapping modality names to their predictions.
                Format: {'CT': {'prediction': 0, 'class_name': 'class0'}, ...}
                If provided, prompts will incorporate this context.
        """
        if not images:
            raise ValueError("LLaVARunner expects at least one image.")
        
        # Use the first available modality
        primary_modality = available_modalities[0] if available_modalities else next(iter(images))
        image = images.get(primary_modality) or next(iter(images.values()))
        
        if isinstance(image, list):
            image = image[0]
        
        return self._predict_single(image, previous_predictions=previous_predictions, current_modality=primary_modality)

