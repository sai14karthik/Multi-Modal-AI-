"""
CLIP-based model wrapper for medical image classification.
Uses zero-shot classification with text prompts.
"""

import contextlib
import numpy as np
from typing import Dict, List, Optional

from src.utils.tf_mock import ensure_tensorflow_stub

ensure_tensorflow_stub()

from PIL import Image, ImageEnhance, ImageOps
import torch
from transformers import CLIPProcessor, CLIPModel


class MultimodalModelWrapper:
    """
    CLIP model wrapper for sequential modality evaluation.
    Performs binary classification for two medical classes.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            model_name: HuggingFace model name for CLIP. Recommended models:
                - "openai/clip-vit-large-patch14" (default, good balance)
                - "openai/clip-vit-base-patch32" (faster, smaller)
                - "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" (larger, better performance)
                - "laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k" (multilingual)
                - "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" (medical-specific, if available)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.is_biomedclip = "biomedclip" in model_name.lower() or "biomed" in model_name.lower()
        self.class_names = class_names or ["Healthy", "Tumor"]
        if len(self.class_names) != 2:
            raise ValueError("MultimodalModelWrapper currently supports exactly two classes.")
        self.class_names = [name.strip() for name in self.class_names]
        self._default_classes = {name.lower() for name in self.class_names} == {"healthy", "tumor"}
        
        print(f"\nLoading model: {model_name}...")
        
        import sys
        import os
        
        # Suppress stderr during model loading to hide verbose messages
        @contextlib.contextmanager
        def suppress_stderr():
            with open(os.devnull, 'w') as devnull:
                old_stderr = sys.stderr
                try:
                    sys.stderr = devnull
                    yield
                finally:
                    sys.stderr = old_stderr
        
        try:
            # Try to load the specified model
            if "biomedclip" in model_name.lower() or "biomed" in model_name.lower():
                loaded = False
                
                try:
                    with suppress_stderr():
                        self.model = CLIPModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
                        self.processor = CLIPProcessor.from_pretrained(model_name, trust_remote_code=True)
                    loaded = True
                except Exception:
                    pass
                
                if not loaded:
                    try:
                        from transformers import AutoModel, AutoProcessor
                        with suppress_stderr():
                            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
                            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                        loaded = True
                    except Exception:
                        pass
                
                if not loaded:
                    try:
                        from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor
                        with suppress_stderr():
                            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
                            try:
                                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                            except Exception:
                                # If AutoProcessor fails, create a CombinedProcessor
                                image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
                                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                                
                                class CombinedProcessor:
                                    def __init__(self, image_processor, tokenizer):
                                        self.image_processor = image_processor
                                        self.tokenizer = tokenizer
                                    def __call__(self, text=None, images=None, return_tensors=None, padding=None, **kwargs):
                                        result = {}
                                        if images is not None:
                                            result.update(self.image_processor(images, return_tensors=return_tensors))
                                        if text is not None:
                                            text_result = self.tokenizer(text, return_tensors=return_tensors, padding=padding, **kwargs)
                                            result.update(text_result)
                                        return result
                                
                                self.processor = CombinedProcessor(image_processor, tokenizer)
                        loaded = True
                    except Exception:
                        pass
                
                if not loaded:
                    self.model_name = "openai/clip-vit-large-patch14"
                    with suppress_stderr():
                        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                        self.processor = CLIPProcessor.from_pretrained(self.model_name)
            elif "laion" in model_name.lower():
                # LAION CLIP models might need different loading
                try:
                    with suppress_stderr():
                        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
                        self.processor = CLIPProcessor.from_pretrained(model_name)
                except Exception as e:
                    print(f"Warning: Could not load {model_name} as CLIPModel, trying AutoModel...")
                    try:
                        from transformers import AutoModel, AutoProcessor
                        with suppress_stderr():
                            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
                            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                    except Exception as e2:
                        print(f"Error loading {model_name}: {e2}")
                        raise
            else:
                with suppress_stderr():
                    self.model = CLIPModel.from_pretrained(model_name).to(self.device)
                    self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to CLIP ViT-Large...")
            self.model_name = "openai/clip-vit-large-patch14"
            try:
                with suppress_stderr():
                    self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                    self.processor = CLIPProcessor.from_pretrained(self.model_name)
            except Exception as fallback_error:
                print(f"Error with fallback: {fallback_error}")
                print("Falling back to CLIP ViT-Base...")
                self.model_name = "openai/clip-vit-base-patch32"
                try:
                    with suppress_stderr():
                        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                        self.processor = CLIPProcessor.from_pretrained(self.model_name)
                except Exception as final_error:
                    raise RuntimeError(
                        f"Failed to load any CLIP model. Original error: {e}, "
                        f"Fallback 1 error: {fallback_error}, Fallback 2 error: {final_error}"
                    ) from final_error
        
        self.model.eval()
        
        # Enhanced prompts with multiple strategies for better performance
        self._init_enhanced_prompts()
    
    def _init_enhanced_prompts(self):
        """Initialize diverse prompt strategies for zero-shot classification."""
        first_class, second_class = self.class_names

        if self._default_classes:
            # Strategy 1: Direct descriptive prompts (lung-specific)
            healthy_direct = [
                "a medical lung CT or PET scan showing healthy normal lung tissue with no tumors or abnormalities",
                "a lung imaging scan with normal anatomy and no pathological findings",
                "a healthy lung medical image showing normal tissue structure without disease",
                "a normal lung scan image with no masses, lesions, or tumors visible"
            ]
            tumor_direct = [
                "a medical lung CT or PET scan showing a visible lung tumor or malignant mass",
                "a lung imaging scan with abnormal mass, tumor growth, or cancerous lesion",
                "a lung medical image showing pathology, tumor, or abnormal tissue",
                "a lung scan image with visible tumor, mass, or pathological abnormality"
            ]
            
            # Strategy 2: Clinical terminology (lung-specific)
            healthy_clinical = [
                "a lung scan with normal lung parenchyma and no abnormal findings",
                "a medical lung image showing normal pulmonary anatomy without pathology",
                "a lung scan demonstrating normal lung tissue architecture"
            ]
            tumor_clinical = [
                "a lung scan with an intrapulmonary mass or neoplasm",
                "a medical lung image showing an abnormal lung lesion or tumor",
                "a lung scan demonstrating pathological lung tissue or mass"
            ]
            
            # Strategy 3: Simple, clear descriptions (lung-specific)
            healthy_simple = [
                "a normal healthy lung scan",
                "a lung scan with no tumor",
                "a healthy lung image"
            ]
            tumor_simple = [
                "a lung scan with a tumor",
                "a lung scan showing a lung tumor",
                "an abnormal lung scan with tumor"
            ]
            
            first_prompts = healthy_direct + healthy_clinical + healthy_simple
            second_prompts = tumor_direct + tumor_clinical + tumor_simple
            prompt_weights = [1.0] * len(healthy_direct) + [0.8] * len(healthy_clinical) + [0.6] * len(healthy_simple)
        else:
            # Detect if this is lung cancer grading (high_grade vs low_grade)
            is_lung_cancer_grading = any(
                keyword in first_class.lower() + second_class.lower()
                for keyword in ["high_grade", "low_grade", "grade", "lung", "nsclc", "adenocarcinoma", "squamous"]
            )
            
            if is_lung_cancer_grading:
                # Lung cancer grade-specific prompts
                first_prompts = [
                    f"a lung CT or PET scan showing {first_class} lung cancer characteristics",
                    f"a lung imaging scan with {first_class} lung cancer features",
                    f"a lung CT or PET slice classified as {first_class} lung cancer",
                    f"a lung cancer scan demonstrating {first_class} tumor grade",
                    f"a lung radiology image with {first_class} lung cancer pathology",
                ]
                second_prompts = [
                    f"a lung CT or PET scan showing {second_class} lung cancer characteristics",
                    f"a lung imaging scan with {second_class} lung cancer features",
                    f"a lung CT or PET slice classified as {second_class} lung cancer",
                    f"a lung cancer scan demonstrating {second_class} tumor grade",
                    f"a lung radiology image with {second_class} lung cancer pathology",
                ]
            else:
                # Generic medical imaging prompts
                context_hint = "medical"
                first_prompts = [
                    f"a {context_hint} imaging scan showing {first_class} characteristics",
                    f"a clinical radiology image labeled as {first_class}",
                    f"a diagnostic slice that is representative of {first_class}",
                ]
                second_prompts = [
                    f"a {context_hint} imaging scan showing {second_class} characteristics",
                    f"a clinical radiology image labeled as {second_class}",
                    f"a diagnostic slice that is representative of {second_class}",
                ]
            prompt_weights = [1.0] * len(first_prompts)
        
        # Create interleaved prompts: [classA1, classB1, classA2, classB2, ...]
        self.class_prompts = [
            prompt for pair in zip(first_prompts, second_prompts)
            for prompt in pair
        ]
        # Interleave weights to match prompt order
        self.weights = [w for pair in zip(prompt_weights, prompt_weights) for w in pair]
    
    def _preprocess_medical_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess medical image for better CLIP performance.
        Enhances contrast and normalizes the image.
        """
        # Convert to grayscale if needed, then back to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast for medical images
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)  # Increase contrast by 30%
        
        # Enhance sharpness slightly
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Apply histogram equalization for better visibility
        # Convert to grayscale for histogram equalization
        gray = np.array(image.convert('L'))
        # Apply histogram equalization
        equalized = ImageOps.equalize(Image.fromarray(gray))
        # Convert back to RGB
        equalized_rgb = Image.new('RGB', equalized.size)
        equalized_rgb.paste(equalized)
        
        # Blend original (70%) with equalized (30%) to preserve natural look
        image = Image.blend(image, equalized_rgb, 0.3)
        
        return image
    
    def predict(
        self,
        images: Dict[str, Image.Image],
        available_modalities: List[str],
        batch_size: int = 1,
        preprocess: bool = False,
        temperature: float = 1.0,
        use_weighted_ensemble: bool = True,
        try_both_swaps: bool = True
    ) -> Dict:
        """
        Predict class using zero-shot classification with enhanced strategies.
        
        Args:
            images: Dictionary mapping modality names to PIL Images (or list of images for batch)
            available_modalities: List of modalities that are available
            batch_size: Batch size for processing (default: 1)
            preprocess: Whether to apply medical image preprocessing (default: False)
            temperature: Temperature scaling for logits (default: 1.0, lower = more confident)
            use_weighted_ensemble: Use weighted average of prompts (default: True)
            try_both_swaps: Try both with and without logit swap, use best (default: True)
        
        Returns:
            Dictionary with prediction (0=Healthy, 1=Tumor), confidence, and probabilities
        """
        available_images = []
        modality_list = []
        for mod in available_modalities:
            if mod in images and images[mod] is not None:
                if isinstance(images[mod], list):
                    available_images.extend(images[mod])
                    modality_list.extend([mod] * len(images[mod]))
                else:
                    available_images.append(images[mod])
                    modality_list.append(mod)
        
        if not available_images:
            raise ValueError(
                f"No valid images found. Available modalities: {available_modalities}, "
                f"Images dict keys: {list(images.keys())}"
            )
        
        # Preprocess images if requested
        if preprocess:
            available_images = [self._preprocess_medical_image(img) for img in available_images]
        
        all_healthy_logits_weighted = []
        all_tumor_logits_weighted = []
        all_healthy_logits_swapped = []
        all_tumor_logits_swapped = []
        
        # Process in batches
        for i in range(0, len(available_images), batch_size):
            batch_images = available_images[i:i + batch_size]
            batch_modalities = modality_list[i:i + batch_size]
            
            inputs = self.processor(
                text=self.class_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                
                # Handle both single image (1D) and batch (2D) cases
                if logits_per_image.dim() == 1:
                    logits_per_image = logits_per_image.unsqueeze(0)
                
                for j in range(len(batch_images)):
                    # Get logits for healthy and tumor prompts
                    # Prompts are interleaved: [healthy1, tumor1, healthy2, tumor2, ...]
                    healthy_logits_all = logits_per_image[j][::2]  # All healthy prompt logits
                    tumor_logits_all = logits_per_image[j][1::2]   # All tumor prompt logits
                    
                    if use_weighted_ensemble and hasattr(self, 'weights'):
                        # Weighted average based on prompt quality
                        healthy_weights = torch.tensor(self.weights[::2], device=healthy_logits_all.device)
                        tumor_weights = torch.tensor(self.weights[1::2], device=tumor_logits_all.device)
                        healthy_prompt_logits = (healthy_logits_all * healthy_weights).sum() / healthy_weights.sum()
                        tumor_prompt_logits = (tumor_logits_all * tumor_weights).sum() / tumor_weights.sum()
                    else:
                        # Simple mean
                        healthy_prompt_logits = healthy_logits_all.mean()
                        tumor_prompt_logits = tumor_logits_all.mean()
                    
                    # Strategy 1: Direct (no swap)
                    all_healthy_logits_weighted.append(healthy_prompt_logits)
                    all_tumor_logits_weighted.append(tumor_prompt_logits)
                    
                    # Strategy 2: Swapped (original behavior)
                    all_healthy_logits_swapped.append(tumor_prompt_logits)
                    all_tumor_logits_swapped.append(healthy_prompt_logits)
        
        # Aggregate across images
        if len(all_healthy_logits_weighted) == 1:
            healthy_logits_direct = all_healthy_logits_weighted[0]
            tumor_logits_direct = all_tumor_logits_weighted[0]
            healthy_logits_swap = all_healthy_logits_swapped[0]
            tumor_logits_swap = all_tumor_logits_swapped[0]
        else:
            healthy_logits_direct = torch.stack(all_healthy_logits_weighted).mean()
            tumor_logits_direct = torch.stack(all_tumor_logits_weighted).mean()
            healthy_logits_swap = torch.stack(all_healthy_logits_swapped).mean()
            tumor_logits_swap = torch.stack(all_tumor_logits_swapped).mean()
        
        # BiomedCLIP works better with direct strategy (no swap)
        # For other models, try both and pick best
        if self.is_biomedclip or not try_both_swaps:
            # Use direct strategy (no swap)
            class_logits = torch.stack([healthy_logits_direct, tumor_logits_direct]) / temperature
            probs = class_logits.softmax(dim=-1)
            healthy_logits = healthy_logits_direct
            tumor_logits = tumor_logits_direct
        else:
            # Try both strategies and pick the one with higher confidence
            # Direct strategy
            class_logits_direct = torch.stack([healthy_logits_direct, tumor_logits_direct]) / temperature
            probs_direct = class_logits_direct.softmax(dim=-1)
            confidence_direct = probs_direct.max().item()
            
            # Swapped strategy
            class_logits_swap = torch.stack([healthy_logits_swap, tumor_logits_swap]) / temperature
            probs_swap = class_logits_swap.softmax(dim=-1)
            confidence_swap = probs_swap.max().item()
            
            # Use the strategy with higher confidence
            # IMPORTANT: If confidences are equal (which is expected for binary classification),
            # prefer the DIRECT strategy. Only swap if swap is STRICTLY more confident.
            if confidence_swap > confidence_direct:
                probs = probs_swap
                healthy_logits = healthy_logits_swap
                tumor_logits = tumor_logits_swap
            else:
                probs = probs_direct
                healthy_logits = healthy_logits_direct
                tumor_logits = tumor_logits_direct
        
        prediction = probs.argmax().item()
        confidence = probs.max().item()
        
        prob_dict = {
            self.class_names[0].lower(): probs[0].item(),
            self.class_names[1].lower(): probs[1].item()
        }
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': prob_dict
        }

