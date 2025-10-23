from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_name: str = "clip_model"):
    from pathlib import Path

    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    clip = CLIP(vision_encoder, text_encoder)
    clip = PeftModel.from_pretrained(clip, model_path).to(device)

    clip.model.load_pretrained(model_path)
    clip.model.eval()
    if device == "cuda":
        clip = clip.to(dtype=torch.bfloat16)

    return clip


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training. Tolerates missing input_ids by
    falling back to labels when necessary.
    """
    # Select text tensors (prefer input_ids, fallback to labels)
    text_tensors = []
    attn_masks = []
    for f in features:
        text_ids = f.get("input_ids", f.get("labels"))
        if text_ids is None:
            raise KeyError("Neither 'input_ids' nor 'labels' found in a feature.")
        text_tensors.append(text_ids)
        if "attention_mask" in f:
            attn_masks.append(f["attention_mask"])
        else:
            attn_masks.append(torch.ones_like(text_ids, dtype=torch.long))

    # Get max sequence length
    max_length = max(t.shape[0] for t in text_tensors)

    def pad_tensor(tensor, pad_value):
        if tensor.shape[0] == max_length:
            return tensor
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(t, pad_value=processor.tokenizer.eos_token_id) for t in text_tensors])
    attention_mask = torch.stack([pad_tensor(m, pad_value=0) for m in attn_masks])
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape
    labels = torch.stack([pad_tensor(f.get("labels", t), pad_value=-100) for f, t in zip(features, text_tensors)])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        self.image_processor = tv.transforms.Compose(
            [
                tv.transforms.Resize(192),
                tv.transforms.RandomResizedCrop(192, scale=(0.5, 1.0)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # placeholder to fit the collator
        }


class CLIP(nn.Module):
    def __init__(
        self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 64, temperature: float = 0.07
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        # TODO: implement the rest components
        vision_hidden = self.vision_encoder.config.hidden_size
        text_hidden = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_hidden, proj_dim, bias=True)
        self.text_proj   = nn.Linear(text_hidden,   proj_dim, bias=True)

        nn.init.normal_(self.vision_proj.weight, std=vision_hidden ** -0.5)
        nn.init.zeros_(self.vision_proj.bias)
        nn.init.normal_(self.text_proj.weight, std=text_hidden ** -0.5)
        nn.init.zeros_(self.text_proj.bias)

        self.register_buffer(
            "logit_log_temp",
            torch.log(torch.tensor(1.0 / float(temperature), dtype=torch.float32))
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(image)

    def encode_text(self, text: str) -> torch.Tensor:
        return self.text_encoder(text)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Customize save method, save additional parameters"""

        additional_state_dict = {}
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data

        torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        """Customize load method, load projection additional parameters"""

        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists():
            additional_state_dict = torch.load(additional_weights_path, map_location="cpu")

            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                param.data = additional_state_dict[name]

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the vision and text backbones.
        (You don't need to touch this method)
        """
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        """
        Enable input require grads for the vision and text backbones.
        (You don't need to touch this method)
        """

        # Reference: https://discuss.huggingface.co/t/peft-lora-gpt-neox-backward-pass-failing/35641
        def make_inputs_require_grads(module, input, output):  # noqa: A002
            output.requires_grad_(True)

        self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CLIP model with improved accuracy.
        
        Args:
            pixel_values: The pixel values of the image (B, C, H, W)
            input_ids: The input ids of the text (B, L)
            attention_mask: The attention mask of the text (B, L) 
            labels: The labels for the text features (compatibility only)
            
        Returns:
            Tuple of (image_features, text_features, logits)
        """
        # Input validation
        if pixel_values.dim() != 4:
            raise ValueError(f"Expected pixel_values to have 4 dimensions, got {pixel_values.dim()}")
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids to have 2 dimensions, got {input_ids.dim()}")
            
        batch_size = pixel_values.shape[0]
        if batch_size != input_ids.shape[0]:
            raise ValueError(f"Batch size mismatch: pixel_values {batch_size} vs input_ids {input_ids.shape[0]}")

        # Vision encoding
        vision_outputs = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        vision_hidden_states = vision_outputs.last_hidden_state  # (B, N, D_vision)
        
        # Improved vision pooling: use CLS token if available, otherwise mean pool
        if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
            vision_pooled = vision_outputs.pooler_output  # (B, D_vision)
        else:
            # Mean pooling over spatial dimension (more stable than global mean)
            vision_pooled = vision_hidden_states.mean(dim=1)  # (B, D_vision)

        # Text encoding  
        text_outputs = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            return_dict=True
        )
        text_hidden_states = text_outputs.last_hidden_state  # (B, L, D_text)

        # Improved text pooling with proper attention mask handling
        if attention_mask is not None:
            # Expand mask to match hidden states dimensions: (B, L) -> (B, L, D_text)
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(text_hidden_states)
            # Apply mask: set padding tokens to 0
            masked_text_states = text_hidden_states * expanded_mask
            # Sum over sequence dimension and divide by number of non-padding tokens
            text_pooled = masked_text_states.sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1e-9)
        else:
            text_pooled = text_hidden_states.mean(dim=1)

        # Ensure consistent dtypes for projection layers
        vision_pooled = vision_pooled.to(self.vision_proj.weight.dtype)
        text_pooled = text_pooled.to(self.text_proj.weight.dtype)

        # Project to shared embedding space
        image_features = self.vision_proj(vision_pooled)  # (B, proj_dim)
        text_features = self.text_proj(text_pooled)      # (B, proj_dim)

        # L2 normalization with improved numerical stability
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-8)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)

        # Temperature-scaled similarity with improved numerical stability
        logit_scale = self.logit_log_temp.to(image_features.dtype).exp()
        logit_scale = logit_scale.clamp(min=1.0, max=100.0)
        logits = logit_scale * torch.matmul(image_features, text_features.t())

        return image_features, text_features, logits

# (moved forward into CLIP class)


def compute_clip_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    image_features, text_features, logits = outputs
    
    if logits.dim() != 2:
        raise ValueError(f"Expected logits to have 2 dimensions, got {logits.dim()}")
    
    batch_size = logits.shape[0]
    if logits.shape[1] != batch_size:
        raise ValueError(f"Expected square logits matrix, got shape {logits.shape}")
    
    if not torch.isfinite(logits).all():
        raise ValueError("Logits contain NaN or infinite values")
    
    device = logits.device
    targets = torch.arange(batch_size, device=device, dtype=torch.long)
    label_smoothing = 0.1
    
    loss_image_to_text = nn.functional.cross_entropy(
        logits, 
        targets, 
        label_smoothing=label_smoothing,
        reduction='mean'
    )
    loss_text_to_image = nn.functional.cross_entropy(
        logits.t(), 
        targets, 
        label_smoothing=label_smoothing,
        reduction='mean'
    )
    total_loss = 0.5 * (loss_image_to_text + loss_text_to_image)
    
    return total_loss


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    target_modules = []
    for name, module in model.named_modules():
        # if isinstance(module, nn.Linear) and ("vision_encoder" in name and "projection" not in name):
        if (
            isinstance(module, nn.Linear)
            and ("vision_encoder" in name or "text_encoder" in name)
            and "projection" not in name
        ):
            target_modules.append(name)

    return target_modules


def train(
    data_dir: Path | None = None,
    output_dir: str = "clip",
    num_train_epochs: float = 0.05,  # for debugging purpose, increase this once the dry run works
    per_device_train_batch_size: int = 1024,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-4,
    num_workers: int = 16,
):
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and processor
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    model = CLIP(vision_encoder, text_encoder).to(device).bfloat16()
    model.set_trainable_parameters()

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        # target_modules="all-linear",
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # load dataset
    train_dataset = CaptionDataset("train", data_dir)
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        bf16=True if device == "cuda" else False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
    )

    trainer.train()

    # save model
    trainer.save_model(output_dir)
    model.model.save_pretrained(output_dir)

    writer.close()

    return model, processor


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_clip",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
    )


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)

    clip = load(ckpt_path)
    clip = clip.model.to(device)

    image_processor = tv.transforms.Compose(
        [
            tv.transforms.Resize(192),
            tv.transforms.CenterCrop(192),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    correct_count = 0
    total_count = 0

    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device).bfloat16()
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if prediction == pair["correct_index"]:
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count}")


def main():
    from fire import Fire

    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()
