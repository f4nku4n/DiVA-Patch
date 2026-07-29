import math

import numpy as np
import torch


def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)


def to_pytorch(tensor_image):
    return torch.from_numpy(tensor_image).permute(2, 0, 1)


def _torch_input(img, unormalize, device):
    img_ = img * 255.0 if unormalize else img
    return to_pytorch(img_)[None, :].to(device)


def _patched_image(base_image, patch, location, device, clip_bounds):
    image = base_image.clone()
    patch_tensor = to_pytorch(patch).to(
        device=device,
        dtype=base_image.dtype,
    )
    x, y = int(location[0]), int(location[1])
    patch_h, patch_w = patch_tensor.shape[-2:]
    image[:, :, x:x + patch_h, y:y + patch_w] = patch_tensor[None, :]
    if clip_bounds is not None:
        image.clamp_(min=clip_bounds[0], max=clip_bounds[1])
    return image


class UnTargeted:
    def __init__(self, model, true, unormalize=False, to_pytorch=False, device='cpu'):
        self.model = model
        self.true = true
        self.unormalize = unormalize
        self.to_pytorch = to_pytorch
        self.device = device
        self.base_image = None

    def bind_base_image(self, image):
        if not self.to_pytorch:
            raise RuntimeError("GPU image caching requires to_pytorch=True")
        self.base_image = _torch_input(image, self.unormalize, self.device)

    def evaluate_patch(self, patch, location, clip_bounds=None):
        if self.base_image is None:
            raise RuntimeError("bind_base_image() must be called before evaluate_patch()")
        patch_ = patch * 255.0 if self.unormalize else patch
        bounds = None
        if clip_bounds is not None:
            bounds = tuple(
                bound * 255.0 if self.unormalize else bound
                for bound in clip_bounds
            )
        image = _patched_image(
            self.base_image, patch_, location, self.device, bounds
        )
        return self._score(self.model.predict(image).flatten())

    def get_label(self, img):
        if self.to_pytorch:
            preds = self.model.predict(
                _torch_input(img, self.unormalize, self.device)
            ).flatten()
            return int(torch.argmax(preds).item())

        img_ = img * 255.0 if self.unormalize else img
        img_ = img_.to(self.device)
        preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
        return int(np.argmax(preds))

    def __call__(self, img):
        if self.to_pytorch:
            preds = self.model.predict(
                _torch_input(img, self.unormalize, self.device)
            ).flatten()
            return self._score(preds)
        else:
            img_ = img * 255.0 if self.unormalize else img
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
        loss = (
            torch.logaddexp(true_logit, log_epsilon)
            - torch.logaddexp(other_logits.max(), log_epsilon)
        )

        # Transfer only the label and scalar loss, with one GPU synchronization.
        y, loss = torch.stack((y.to(preds.dtype), loss)).tolist()
        return [int(y) != self.true, float(loss)]


class Targeted:
    def __init__(self, model, true, target, unormalize=False, to_pytorch=False, device='cpu'):
        self.model = model
        self.true = true
        self.target = target
        self.unormalize = unormalize
        self.to_pytorch = to_pytorch
        self.device = device
        self.base_image = None

    def bind_base_image(self, image):
        if not self.to_pytorch:
            raise RuntimeError("GPU image caching requires to_pytorch=True")
        self.base_image = _torch_input(image, self.unormalize, self.device)

    def evaluate_patch(self, patch, location, clip_bounds=None):
        if self.base_image is None:
            raise RuntimeError("bind_base_image() must be called before evaluate_patch()")
        patch_ = patch * 255.0 if self.unormalize else patch
        bounds = None
        if clip_bounds is not None:
            bounds = tuple(
                bound * 255.0 if self.unormalize else bound
                for bound in clip_bounds
            )
        image = _patched_image(
            self.base_image, patch_, location, self.device, bounds
        )
        return self._score(self.model.predict(image).flatten())

    def get_label(self, img):
        if self.to_pytorch:
            preds = self.model.predict(
                _torch_input(img, self.unormalize, self.device)
            ).flatten()
            return int(torch.argmax(preds).item())

        img_ = img * 255.0 if self.unormalize else img
        img_ = img_.to(self.device)
        preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
        return int(np.argmax(preds))

    def __call__(self, img):
        if self.to_pytorch:
            preds = self.model.predict(
                _torch_input(img, self.unormalize, self.device)
            ).flatten()
            return self._score(preds)
        else:
            img_ = img * 255.0 if self.unormalize else img
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
