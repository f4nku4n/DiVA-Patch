import torch
from torchvision import models as torch_models
import torchvision.transforms.functional as F

def preprocess_for_inference(tensor):
    tensor = F.resize(tensor, size=256, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
    tensor = F.center_crop(tensor, 224)
    tensor = tensor.float() / 255.0
    return tensor


class ImageNetModel:
    def __init__(self, model: int, device='cpu'):
        model_class_dict = [torch_models.vgg16_bn, torch_models.resnet50, torch_models.vit_b_16]
        # model_pt = model_class_dict[model](pretrained=True)
        model_pt = model_class_dict[model](pretrained=False)
        if model == 0:
            state_dict = torch.load('/kaggle/input/models/quanphanminhdede/diva-patch-models/pytorch/default/1/vgg16_bn-6c64b313.pth', weights_only=True)
        elif model == 1:
            state_dict = torch.load('/kaggle/input/models/quanphanminhdede/diva-patch-models/pytorch/default/1/resnet50-0676ba61.pth', weights_only=True)
        elif model == 2:
            state_dict = torch.load('/kaggle/input/models/quanphanminhdede/diva-patch-models/pytorch/default/1/vit_b_16-c867db91.pth', weights_only=True)
        else:
            raise ValueError
        model_pt.load_state_dict(state_dict)

        self.model = model_pt.to(device)
        self.model.eval()

        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).to(device)
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).to(device)

    @torch.inference_mode()
    def predict(self, x):
        x = preprocess_for_inference(x)
        out = (x - self.mu) / self.sigma
        return self.model(out)

    def forward(self, x):
        x = preprocess_for_inference(x)
        out = (x - self.mu) / self.sigma
        return self.model(out)

    def __call__(self, x):
        return self.predict(x)


class RNDImageNet:
    def __init__(self, idx):
        self.model = ImageNetModel(idx)
        self.v = 0.02

    def predict(self, x):
        #x_ = x + np.random.normal(0, 1, size=x.shape) * self.v
        x_ = x + torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x)) * self.v
        return self.model.predict(x_)
