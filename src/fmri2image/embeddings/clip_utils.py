import torch
import numpy as np
from typing import Tuple
import open_clip


def load_openclip(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str = None):
    """
    Load OpenCLIP model, tokenizer and preprocess.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, tokenizer, preprocess, device


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device="cuda", batch_size=256, normalize=True) -> np.ndarray:
    """
    Encode a list of strings into CLIP text embeddings.
    """
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        feats = model.encode_text(tokens)
        feats = feats.float()
        if normalize:
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        embs.append(feats.cpu().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def encode_images(model, preprocess, pil_images, device="cuda", batch_size=64, normalize=True) -> np.ndarray:
    """
    Encode a list of PIL images into CLIP image embeddings.
    """
    import torch
    import numpy as np
    tensors = [preprocess(img) for img in pil_images]
    embs = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i+batch_size]).to(device)
        feats = model.encode_image(batch).float()
        if normalize:
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        embs.append(feats.cpu().numpy())
    return np.concatenate(embs, axis=0)


def save_embeddings_npy_pkl(npy_path: str, pkl_path: str, embeddings: np.ndarray, meta: dict):
    """
    Save embeddings to .npy and metadata to .pkl for later checks.
    """
    import pickle, os
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    np.save(npy_path, embeddings)
    with open(pkl_path, "wb") as f:
        pickle.dump(meta, f)
