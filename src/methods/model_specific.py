"""
Model-Specific XAI Methods — Grad-CAM for CNNs.

Grad-CAM computes the gradient of the target class score with respect
to the feature maps of the last convolutional layer, then creates a
weighted heatmap highlighting influential spatial regions.
"""

from __future__ import annotations

import numpy as np


def grad_cam(model, input_tensor, target_class: int | None = None):
    """
    Compute Grad-CAM heatmap for a PyTorch CNN.

    Parameters
    ----------
    model : nn.Module
        A PyTorch CNN that has a `.features` sequential block.
    input_tensor : torch.Tensor
        Shape (1, C, H, W).
    target_class : int or None
        Class index to explain. If None, uses the predicted class.

    Returns
    -------
    heatmap : np.ndarray
        2-D heatmap (H', W') normalised to [0, 1].
    predicted_class : int
    """
    import torch
    import torch.nn.functional as F

    model.eval()

    # --- Hook to capture the last conv layer's activations & gradients ---
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    # Register hooks on the last conv layer inside model.features
    last_conv = None
    for module in model.features.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model.features")

    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Backward pass for the target class
    model.zero_grad()
    score = output[0, target_class]
    score.backward()

    # Remove hooks
    fh.remove()
    bh.remove()

    # Grad-CAM computation
    grads = gradients["value"]            # (1, K, H', W')
    acts = activations["value"]           # (1, K, H', W')

    # α_k = global-average-pooling of gradients
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

    # Weighted combination
    cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
    cam = F.relu(cam)                                # ReLU
    cam = cam.squeeze().detach().cpu().numpy()

    # Normalise to [0, 1]
    if cam.max() != cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    return cam, target_class


def grad_cam_overlay(original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5):
    """
    Overlay the Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_image : np.ndarray   (H, W) or (H, W, C) in [0, 1].
    heatmap        : np.ndarray   (H', W') in [0, 1].
    alpha          : blend factor.

    Returns
    -------
    overlay : np.ndarray  (H, W, 3) in [0, 1].
    """
    import cv2

    # Resize heatmap to original image size
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to colour map (jet)
    heatmap_colour = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colour = heatmap_colour.astype(np.float32) / 255.0

    # Ensure original is 3-channel
    if original_image.ndim == 2:
        original_3ch = np.stack([original_image] * 3, axis=-1)
    else:
        original_3ch = original_image

    overlay = alpha * heatmap_colour + (1 - alpha) * original_3ch
    overlay = np.clip(overlay, 0, 1)
    return overlay
