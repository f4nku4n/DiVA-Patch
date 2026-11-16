import math

import numpy as np
import torch

from utils.BatchEvaluator import PatchQuery


def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)


def convert_to_pytorch(tensor_image):
    return torch.from_numpy(tensor_image).permute(2, 0, 1)


def _torch_input(img, unNormalized, device):
    img_ = img * 255.0 if unNormalized else img
    return convert_to_pytorch(img_)[None, :].to(device)


def _patched_image(base_image, patch, location, device, clip_bounds):
    image = base_image.clone()
    patch_tensor = convert_to_pytorch(patch).to(device=device, dtype=base_image.dtype)
    x, y = int(location[0]), int(location[1])
    patch_h, patch_w = patch_tensor.shape[-2:]
    image[:, :, x:x + patch_h, y:y + patch_w] = patch_tensor[None, :]
    if clip_bounds is not None:
        image.clamp_(min=clip_bounds[0], max=clip_bounds[1])
    return image


def _rectangle_image(base_image, target_image, rectangle):
    top, left, bottom, right = (int(value) for value in rectangle)
    image = base_image.clone()
    image[:, :, top:bottom, left:right] = target_image[:, :, top:bottom, left:right]
    return image


class UnTargeted:
    def __init__(self, model, true, unNormalized=False, to_pytorch=False, device='cpu'):
        self.model = model
        self.true = true
        self.unNormalized = unNormalized
        self.to_pytorch = to_pytorch
        self.device = device
        self.base_image = None
        self.target_image = None

    def bind_base_image(self, image):
        if not self.to_pytorch:
            raise RuntimeError("GPU image caching requires to_pytorch=True")
        self.base_image = _torch_input(image, self.unNormalized, self.device)

    def bind_target_image(self, image):
        if not self.to_pytorch:
            raise RuntimeError("GPU image caching requires to_pytorch=True")
        self.target_image = _torch_input(image, self.unNormalized, self.device)
        if self.base_image is not None and self.target_image.shape != self.base_image.shape:
            raise ValueError("base and target images must have the same shape")

    def evaluate_rectangle(self, rectangle):
        if self.base_image is None or self.target_image is None:
            raise RuntimeError("Bind_base_image() and bind_target_image() must be called first")
        image = _rectangle_image(self.base_image, self.target_image, rectangle)
        prediction = torch.argmax(self.model.predict(image).flatten())
        return bool(int(prediction.item()) != self.true)

    def evaluate_patch(self, patch, location, clip_bounds=None):
        if self.base_image is None:
            raise RuntimeError("Bind_base_image() must be called before evaluate_patch()")
        patch_ = patch * 255.0 if self.unNormalized else patch
        bounds = None
        if clip_bounds is not None:
            bounds = tuple(bound * 255.0 if self.unNormalized else bound for bound in clip_bounds)
        image = _patched_image(self.base_image, patch_, location, self.device, bounds)
        return self._score(self.model.predict(image).flatten())

    def make_patch_query(self, patch, location, clip_bounds=None):
        if self.base_image is None:
            raise RuntimeError("Bind_base_image() must be called before make_patch_query()")
        return PatchQuery(
            base_image=self.base_image,
            patch=patch,
            location=tuple(location),
            targeted=False,
            true_label=int(self.true),
            unnormalized=self.unNormalized,
            clip_bounds=clip_bounds,
        )

    def get_label(self, img):
        if self.to_pytorch:
            preds = self.model.predict(_torch_input(img, self.unNormalized, self.device)).flatten()
            return int(torch.argmax(preds).item())

        img_ = img * 255.0 if self.unNormalized else img
        img_ = img_.to(self.device)
        preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
        return int(np.argmax(preds))

    def __call__(self, img):
        if self.to_pytorch:
            preds = self.model.predict(_torch_input(img, self.unNormalized, self.device)).flatten()
            return self._score(preds)
        else:
            img_ = img * 255.0 if self.unNormalized else img
            img_ = img_.to(self.device)
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

            f_true = math.log(math.exp(preds[self.true]) + 1e-30)
            preds[self.true] = -math.inf
            f_other = math.log(math.exp(max(preds)) + 1e-30)
            loss = f_true - f_other

        return [y != self.true, float(loss)]

    def _score(self, preds):
        y = torch.argmax(preds)
        true_logit = preds[self.true]
        other_logits = preds.clone()
        other_logits[self.true] = -torch.inf
        log_epsilon = preds.new_tensor(math.log(1e-30))
        loss = torch.logaddexp(true_logit, log_epsilon) - torch.logaddexp(other_logits.max(), log_epsilon)

        # Transfer only the label and scalar loss, with one GPU synchronization.
        y, loss = torch.stack((y.to(preds.dtype), loss)).tolist()
        return [int(y) != self.true, float(loss)]


class Targeted:
    def __init__(self, model, true, target, unNormalized=False, to_pytorch=False, device='cpu'):
        self.model = model
        self.true = true
        self.target = target
        self.unNormalized = unNormalized
        self.to_pytorch = to_pytorch
        self.device = device
        self.base_image = None
        self.target_image = None

    def bind_base_image(self, image):
        if not self.to_pytorch:
            raise RuntimeError("GPU image caching requires to_pytorch=True")
        self.base_image = _torch_input(image, self.unNormalized, self.device)

    def bind_target_image(self, image):
        if not self.to_pytorch:
            raise RuntimeError("GPU image caching requires to_pytorch=True")
        self.target_image = _torch_input(image, self.unNormalized, self.device)
        if self.base_image is not None and self.target_image.shape != self.base_image.shape:
            raise ValueError("base and target images must have the same shape")

    def evaluate_rectangle(self, rectangle):
        if self.base_image is None or self.target_image is None:
            raise RuntimeError(
                "bind_base_image() and bind_target_image() must be called first"
            )
        image = _rectangle_image(self.base_image, self.target_image, rectangle)
        prediction = torch.argmax(self.model.predict(image).flatten())
        return bool(int(prediction.item()) == self.target)

    def evaluate_patch(self, patch, location, clip_bounds=None):
        if self.base_image is None:
            raise RuntimeError("bind_base_image() must be called before evaluate_patch()")
        patch_ = patch * 255.0 if self.unNormalized else patch
        bounds = None
        if clip_bounds is not None:
            bounds = tuple(bound * 255.0 if self.unNormalized else bound for bound in clip_bounds)
        image = _patched_image(self.base_image, patch_, location, self.device, bounds)
        return self._score(self.model.predict(image).flatten())

    def make_patch_query(self, patch, location, clip_bounds=None):
        if self.base_image is None:
            raise RuntimeError("bind_base_image() must be called before make_patch_query()")
        return PatchQuery(
            base_image=self.base_image,
            patch=patch,
            location=tuple(location),
            targeted=True,
            true_label=int(self.true),
            target_label=int(self.target),
            unnormalized=self.unNormalized,
            clip_bounds=clip_bounds,
        )

    def get_label(self, img):
        if self.to_pytorch:
            preds = self.model.predict(_torch_input(img, self.unNormalized, self.device)).flatten()
            return int(torch.argmax(preds).item())

        img_ = img * 255.0 if self.unNormalized else img
        img_ = img_.to(self.device)
        preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
        return int(np.argmax(preds))

    def __call__(self, img):
        if self.to_pytorch:
            preds = self.model.predict(_torch_input(img, self.unNormalized, self.device)).flatten()
            return self._score(preds)
        else:
            img_ = img * 255.0 if self.unNormalized else img
            img_ = img_.to(self.device)
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

            f_target = preds[self.target]
            f_other = math.log(sum(math.exp(pi) for pi in preds))
            loss = f_other - f_target

        return [y == self.target, float(loss)]

    def _score(self, preds):
        y = torch.argmax(preds)
        loss = torch.logsumexp(preds, dim=0) - preds[self.target]

        # Transfer only the label and scalar loss, with one GPU synchronization.
        y, loss = torch.stack((y.to(preds.dtype), loss)).tolist()
        return [int(y) == self.target, float(loss)]
