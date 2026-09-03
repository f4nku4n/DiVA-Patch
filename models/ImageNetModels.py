import torch
from torchvision import models as torch_models
import torchvision.transforms.functional as F

def preprocess_for_inference(tensor):
    tensor = F.resize(tensor, size=256, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
    tensor = F.center_crop(tensor, 224)
    tensor = tensor.float() / 255.0
    return tensor


class ImageNetModel:
    def __init__(self, model: int, device='cpu', setting='realistic'):
        model_class_dict = [torch_models.vgg16_bn, torch_models.resnet50, torch_models.vit_b_16]
        model_pt = model_class_dict[model](pretrained=True)
        # root = '/kaggle/input/models/f4nku4n99/weights/pytorch/default/1'
        # root = '/kaggle/input/models/quanphanminhdede/diva-patch-models/pytorch/default/1'
        # model_pt = model_class_dict[model](pretrained=False)
        # if model == 0:
        #     state_dict = torch.load(f'{root}/vgg16_bn-6c64b313.pth', weights_only=True)
        # elif model == 1:
        #     state_dict = torch.load(f'{root}/resnet50-0676ba61.pth', weights_only=True)
        # elif model == 2:
        #     state_dict = torch.load(f'{root}/vit_b_16-c867db91.pth', weights_only=True)
        # else:
        #     raise ValueError
        # model_pt.load_state_dict(state_dict)
        self.setting = setting

        self.model = model_pt.to(device)
        self.model.eval()

        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).to(device)
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).to(device)

    @torch.inference_mode()
    def predict(self, x):
        if self.setting == 'realistic':
            x = preprocess_for_inference(x)
        out = (x - self.mu) / self.sigma
        return self.model(out)

    def forward(self, x):
        if self.setting == 'realistic':
            x = preprocess_for_inference(x)
        out = (x - self.mu) / self.sigma
        return self.model(out)

    def __call__(self, x):
        return self.predict(x)
