import time
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
import vit
from predictor import SamPredictor
import os
import onnxruntime as ort
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
from copy import deepcopy
from typing import Tuple

# run on cuda:1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    for i in range(len(coords)):
        pos_points = coords[labels == i + 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def transform(img: np.ndarray) -> np.ndarray:
    """image transform

    This function can convert the input image to the required input format for vit.

    Args:
        img (np.ndarray): input image, the image type should be BGR.

    Returns:
        np.ndarray: transformed image.
    """
    h, w, c = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - mean) / std

    size = max(h, w)
    img = np.pad(img, ((0, size - h), (0, size - w), (0, 0)), 'constant', constant_values=(0, 0))
    img = cv2.resize(img, input_shape[2:])
    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, axes=[0, 3, 1, 2]).astype(np.float32)
    return img


def apply_coords(coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old_h, old_w = original_size

    scale = img_size[0] * 1.0 / max(original_size[0], original_size[1])
    newh, neww = original_size[0] * scale, original_size[1] * scale
    new_w = int(neww + 0.5)
    new_h = int(newh + 0.5)

    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


# args
vit_model = 'export_onnx/vit_b.onnx'
img_size = (1024, 1024)
mask_threshold = 0.0

img = cv2.imread('../dog.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

vit_session = ort.InferenceSession(vit_model,
                                   ort.SessionOptions(),
                                   providers=['CUDAExecutionProvider'])  # CUDAExecutionProvider

input_name = vit_session.get_inputs()[0].name
input_shape = vit_session.get_inputs()[0].shape
output_name = vit_session.get_outputs()[0].name
output_shape = vit_session.get_outputs()[0].shape

mean = np.array([123.675, 116.28, 103.53])
std = np.array([58.395, 57.12, 57.375])

# prepare data
input_point = np.array([[200, 400]])
input_label = np.array([1])

onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
# onnx_label = input_label[None, :].astype(np.float32)

onnx_coord = apply_coords(onnx_coord, img.shape[:2]).astype(np.float32)

onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

# embeddings
tensor = transform(img)

t0 = time.time()
feature = vit_session.run(None, {input_name: tensor})[0]
t1 = time.time()
print(t1 - t0)
# point input example
onnx_model_path = "../sam_onnx_example.onnx"
ort_session = ort.InferenceSession(onnx_model_path,
                                   providers=['CUDAExecutionProvider'])

ort_inputs = {
    "image_embeddings": feature,
    "point_coords": onnx_coord,
    "point_labels": onnx_label,
    "mask_input": onnx_mask_input,
    "has_mask_input": onnx_has_mask_input,
    "orig_im_size": np.array(img.shape[:2], dtype=np.float32)
}

masks, _, low_res_logits = ort_session.run(None, ort_inputs)

masks = masks > mask_threshold
plt.figure(figsize=(10, 10))
plt.imshow(img)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()
