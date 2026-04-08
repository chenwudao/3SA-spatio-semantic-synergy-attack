from __future__ import annotations

# 必须在所有导入之前设置警告过滤
import contextlib
import io
import os
import warnings

# 抑制 transformers 的 FlashAttention 提示（通过环境变量）
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="FlashAttention2 is not installed")
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import math

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModel, AutoTokenizer

# 设置 Hugging Face 镜像（中国大陆访问优化）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from .base import AttentionOutput, LossOutput, VisionLanguageSurrogate


@contextlib.contextmanager
def _suppress_stdout():
    """Suppress print() output from third-party model code (e.g. InternVL)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class InternVLSurrogate(VisionLanguageSurrogate):
    """
    InternVL Surveillance via Transformers with trust_remote_code.
    Extracts the global vision-encoder self-attention map from InternViT.
    Uses dynamic res blocking, simplified here for stage 1 (1x1 block).
    """
    def __init__(self, model_id: str = "OpenGVLab/InternVL2-1B", weight: float = 1.0, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        name = model_id.split("/")[-1]
        super().__init__(name=name, weight=weight)
        self.device = device

        # InternVL typically needs trust_remote_code=True
        # Suppress stdout from InternVL's internal code (FlashAttention2 notice)
        with _suppress_stdout():
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_id,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager",  # Required for output_attentions=True
            ).to(self.device).eval()

        # InternVL normally expects 448x448 squares
        self.transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def extract_attention(self, image: torch.Tensor, text_prompt: str) -> AttentionOutput:
        b, c, h, w = image.shape
        img_np = (image[0].permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(img_np)

        # Single block 448x448 for stage 1 (IoU analysis on standard input)
        pixel_values = self.transform(pil_img).unsqueeze(0).to(self.device, torch.float16)

        with torch.no_grad():
            # InternVL vision_model doesn't support output_attentions,
            # so we use a forward hook to capture attention from the last block
            attention_map = None
            attn_module = self.model.vision_model.encoder.layers[-1].attn
            num_heads = attn_module.num_heads
            scale = attn_module.scale

            def hook_fn(module, input, output):
                nonlocal attention_map
                # output shape: (B, N, 3 * num_heads * head_dim)
                B, N, _ = output.shape
                qkv = output.reshape(B, N, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                # Compute attention: softmax(Q @ K^T / sqrt(d))
                attn = (q @ k.transpose(-2, -1)) * scale
                attention_map = attn.softmax(dim=-1)

            # Hook the qkv linear in the last attention layer
            hook = attn_module.qkv.register_forward_hook(hook_fn)

            vision_outputs = self.model.vision_model(pixel_values=pixel_values)

            # Remove hook
            hook.remove()

            # attention_map shape: (batch, num_heads, seq_len, seq_len)
            last_layer_attn = attention_map
            cls_attn = last_layer_attn[:, :, 0, 1:]  # Drop cls attending to cls
            avg_cls_attn = cls_attn.mean(dim=1)  # average over heads -> (1, num_patches)

            num_patches = avg_cls_attn.shape[1]
            grid_size = int(math.sqrt(num_patches))

            attn_map = avg_cls_attn.reshape(1, 1, grid_size, grid_size).float()

            # Resize back to original dimension (h, w)
            attn_map_resized = torch.nn.functional.interpolate(attn_map, size=(h, w), mode='bilinear', align_corners=False)

            attn_min = attn_map_resized.amin(dim=(2, 3), keepdim=True)
            attn_max = attn_map_resized.amax(dim=(2, 3), keepdim=True)
            attn_norm = (attn_map_resized - attn_min) / (attn_max - attn_min + 1e-8)

        return AttentionOutput(
            attention_map=attn_norm.cpu(),
            metadata={"source": "internvl_vision", "heads_averaged": True}
        )

    def compute_loss(self, image: torch.Tensor, text_prompt: str) -> LossOutput:
        raise NotImplementedError

class LLaVASurrogate(VisionLanguageSurrogate):
    """
    LLaVA Surrogate using 🤗 Transformers.
    Extracts cross-attention or vision-encoder self-attention map.
    Since we care about vision, we extract vision tower's self-attention of CLS to patches.
    """
    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf", weight: float = 1.0, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        name = model_id.split("/")[-1]
        super().__init__(name=name, weight=weight)
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",  # Required for output_attentions=True
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    def extract_attention(self, image: torch.Tensor, text_prompt: str) -> AttentionOutput:
        # Our dataset returns image tensors (b, c, h, w) in [0, 1] mapped from PIL.
        # Llava processor usually takes PIL images, so let's convert back.
        # Alternatively, we can feed pixel values directly, but it's safer to use the processor.
        b, c, h, w = image.shape
        # Assuming batch size = 1 for stage 1.
        img_np = (image[0].permute(1, 2, 0).numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(img_np)
        
        # Format conversation for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt").to(self.device, torch.float16)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )
            
            # Extract vision model's attention.
            # LLaVA uses a vision_tower (CLIP) which outputs attentions if output_attentions=True is configured
            vision_tower = self.model.vision_tower
            vision_outputs = vision_tower(inputs["pixel_values"], output_attentions=True, return_dict=True)
            
            # Get the last layer's self-attention
            last_layer_attn = vision_outputs.attentions[-1] # shape: (batch, num_heads, seq_len, seq_len)
            
            # The vision tokens have a CLS token at index 0. We want attention from CLS to patches.
            cls_attn = last_layer_attn[:, :, 0, 1:] # Drop cls attending to cls
            avg_cls_attn = cls_attn.mean(dim=1) # average over heads: (batch, num_patches)
            
            # In CLIP-ViT-L/14 the patch count is 576 for 336x336 input (24x24)
            num_patches = avg_cls_attn.shape[1]
            grid_size = int(math.sqrt(num_patches))
            
            attn_map = avg_cls_attn.reshape(b, 1, grid_size, grid_size).float()
            
            # Resize back to original
            attn_map_resized = torch.nn.functional.interpolate(attn_map, size=(h, w), mode='bilinear', align_corners=False)
            
            # Normalize to [0,1]
            attn_min = attn_map_resized.amin(dim=(2, 3), keepdim=True)
            attn_max = attn_map_resized.amax(dim=(2, 3), keepdim=True)
            attn_norm = (attn_map_resized - attn_min) / (attn_max - attn_min + 1e-8)

        return AttentionOutput(
            attention_map=attn_norm.cpu(),
            metadata={"source": "llava_vision", "heads_averaged": True}
        )

    def compute_loss(self, image: torch.Tensor, text_prompt: str) -> LossOutput:
        raise NotImplementedError

class DINOv2Surrogate(VisionLanguageSurrogate):
    """
    DINOv2 as a pure vision baseline (no text prompt).
    Extracts attention from the [CLS] token of the final layer.
    """
    def __init__(self, name: str = "dinov2_vits14", weight: float = 1.0, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(name=name, weight=weight)
        self.device = device
        # Load dinov2 directly from torch hub
        self.model = torch.hub.load("facebookresearch/dinov2", name).to(self.device).eval()

        # DINOv2 transform
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Patch size is 14 for vits14
        self.patch_size = 14

    def extract_attention(self, image: torch.Tensor, text_prompt: str = "") -> AttentionOutput:
        """
        Since DINOv2 doesn't use text, prompt is ignored.
        image is assumed to be raw tensor. We apply DINOv2 transforms inside.
        If your dataset already resizes/normalizes, you may bypass this.
        For now, let's normalize exactly for dinov2.
        Wait, `image` from our dataset is a tensor in [0, 1] sized roughly (batch, 3, 336, 336).
        DINOv2 requires 224x224 (or divisible by patch_size 14).
        """
        # Let's ensure tensor is correctly shaped and normalized for DINO
        b, c, h, w = image.shape
        # To make it compatible, we can interpolate to 224x224
        # and then normalize using imagenet means
        img_resized = torch.nn.functional.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_resized.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_resized.device).view(1, 3, 1, 1)
        img_norm = (img_resized - mean) / std

        img_norm = img_norm.to(self.device)

        with torch.no_grad():
            # DINOv2 Attention module doesn't return attention weights,
            # so we hook the qkv output and compute attention manually
            attention_map = None
            attn_module = self.model.blocks[-1].attn  # Capture reference to attention module
            num_heads = attn_module.num_heads
            scale = attn_module.scale

            def hook_fn(module, input, output):
                nonlocal attention_map
                # output shape: (B, N, 3 * num_heads * head_dim)
                B, N, _ = output.shape
                qkv = output.reshape(B, N, 3, num_heads, -1).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                # Compute attention: softmax(Q @ K^T / sqrt(d))
                attn = (q @ k.transpose(-2, -1)) * scale
                attention_map = attn.softmax(dim=-1)

            # Hook the qkv linear layer in the last block's attention
            hook = attn_module.qkv.register_forward_hook(hook_fn)

            # Forward pass
            _ = self.model(img_norm)

            # Remove hook
            hook.remove()

            # attention_map is [batch, num_heads, num_tokens, num_tokens]
            # CLS token is at index 0
            cls_attention = attention_map[:, :, 0, 1:]  # Drop cls self-attention

            # Average across heads
            avg_cls_attention = cls_attention.mean(dim=1)  # [batch, num_tokens-1]

            # Reshape to feature map
            num_patches = 224 // self.patch_size
            attn_map = avg_cls_attention.reshape(b, 1, num_patches, num_patches)

            # Interpolate back to original image size
            attn_map_resized = torch.nn.functional.interpolate(attn_map, size=(h, w), mode='bilinear', align_corners=False)

            # Normalize to [0,1]
            attn_min = attn_map_resized.amin(dim=(2, 3), keepdim=True)
            attn_max = attn_map_resized.amax(dim=(2, 3), keepdim=True)
            attn_norm = (attn_map_resized - attn_min) / (attn_max - attn_min + 1e-8)

        return AttentionOutput(
            attention_map=attn_norm.cpu(),
            metadata={"source": "dinov2", "heads_averaged": True}
        )

    def compute_loss(self, image: torch.Tensor, text_prompt: str) -> LossOutput:
        raise NotImplementedError("Loss calculation for DINO baseline not implemented for stage 1")

