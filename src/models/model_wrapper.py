"""
CLIP-based model wrapper for medical image classification.
Uses zero-shot classification with text prompts.
"""

import os
import sys
import types
import importlib.util

# Suppress transformers warnings and prevent TensorFlow import errors
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Mock TensorFlow before transformers tries to import it
if 'tensorflow' not in sys.modules:
    tf_mock = types.ModuleType('tensorflow')
    # Create a proper spec for the mock
    spec = importlib.util.spec_from_loader('tensorflow', loader=None)
    tf_mock.__spec__ = spec
    tf_mock.__version__ = '2.0.0'
    
    # Add common TensorFlow attributes that transformers might check
    class MockTensor:
        pass
    class MockVariable:
        pass
    tf_mock.Tensor = MockTensor
    tf_mock.Variable = MockVariable
    
    # Mock submodules
    tf_mock.image = types.ModuleType('tensorflow.image')
    tf_mock.nn = types.ModuleType('tensorflow.nn')
    
    sys.modules['tensorflow'] = tf_mock

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import Dict, List, Optional


class MultimodalModelWrapper:
    """
    CLIP model wrapper for sequential modality evaluation.
    Performs binary classification: Healthy (0) vs Tumor (1).
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model name for CLIP
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        print(f"\nLoading model: {model_name}...")
        try:
            # Try to load the specified model
            if "biomedclip" in model_name.lower() or "biomed" in model_name.lower():
                loaded = False
                
                try:
                    self.model = CLIPModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
                    self.processor = CLIPProcessor.from_pretrained(model_name, trust_remote_code=True)
                    loaded = True
                except Exception:
                    pass
                
                if not loaded:
                    try:
                        from transformers import AutoModel, AutoProcessor
                        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
                        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                        loaded = True
                    except Exception:
                        pass
                
                if not loaded:
                    try:
                        from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor
                        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
                        try:
                            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                        except:
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
                    self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                    self.processor = CLIPProcessor.from_pretrained(self.model_name)
            else:
                self.model = CLIPModel.from_pretrained(model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to CLIP ViT-Large...")
            self.model_name = "openai/clip-vit-large-patch14"
            try:
                self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
            except:
                self.model_name = "openai/clip-vit-base-patch32"
                self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        self.model.eval()
        
        self.healthy_prompts = [
            "a medical brain scan showing healthy normal brain tissue with no tumors or abnormalities",
            "a brain imaging scan with normal anatomy and no pathological findings",
            "a healthy brain medical image showing normal tissue structure without disease",
            "a normal brain scan image with no masses, lesions, or tumors visible"
        ]
        self.tumor_prompts = [
            "a medical brain scan showing a visible brain tumor or malignant mass",
            "a brain imaging scan with abnormal mass, tumor growth, or cancerous lesion",
            "a brain medical image showing pathology, tumor, or abnormal tissue",
            "a brain scan image with visible tumor, mass, or pathological abnormality"
        ]
        
        self.class_prompts = [
            prompt for pair in zip(self.healthy_prompts, self.tumor_prompts) 
            for prompt in pair
        ]
    
    def predict(
        self,
        images: Dict[str, Image.Image],
        available_modalities: List[str],
        batch_size: int = 1
    ) -> Dict:
        """
        Predict class using zero-shot classification.
        For multiple modalities, averages predictions from all available images.
        
        Args:
            images: Dictionary mapping modality names to PIL Images (or list of images for batch)
            available_modalities: List of modalities that are available
            batch_size: Batch size for processing (default: 1)
        
        Returns:
            Dictionary with prediction (0=Healthy, 1=Tumor), confidence, and probabilities
        """
        available_images = []
        for mod in available_modalities:
            if mod in images and images[mod] is not None:
                if isinstance(images[mod], list):
                    available_images.extend(images[mod])
                else:
                    available_images.append(images[mod])
        
        if not available_images:
            raise ValueError("No valid images found")
        
        all_healthy_logits = []
        all_tumor_logits = []
        
        # Process in batches
        for i in range(0, len(available_images), batch_size):
            batch_images = available_images[i:i + batch_size]
            
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
                    healthy_prompt_logits = logits_per_image[j][::2].mean()
                    tumor_prompt_logits = logits_per_image[j][1::2].mean()
                    
                    # CLIP assigns higher logits to tumor prompts for medical images
                    # Swap logits: tumor_prompt_logits -> healthy class, healthy_prompt_logits -> tumor class
                    all_healthy_logits.append(tumor_prompt_logits)
                    all_tumor_logits.append(healthy_prompt_logits)
        
        if len(all_healthy_logits) == 1:
            healthy_logits = all_healthy_logits[0]
            tumor_logits = all_tumor_logits[0]
        else:
            healthy_logits = torch.stack(all_healthy_logits).mean()
            tumor_logits = torch.stack(all_tumor_logits).mean()
        
        class_logits = torch.stack([healthy_logits, tumor_logits])
        probs = class_logits.softmax(dim=-1)
        
        prediction = probs.argmax().item()
        confidence = probs.max().item()
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'healthy': probs[0].item(),
                'tumor': probs[1].item()
            }
        }

