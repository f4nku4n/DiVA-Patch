import numpy as np
from PIL import Image
from utils import pytorch_switch
from torchvision import transforms

def getVisionModel(dataset, vision_model):
    if dataset == 'ImageNet1K':
        from models.ImageNetModels import ImageNetModel as VisionModel
        model_args = ({"VGGNet16": 0, "ResNet50": 1, "ViT16": 2}[vision_model],)
    elif dataset == 'Flower102':
        from models.Flower102Models import Flower102Model as VisionModel
        model_args = ()
    elif dataset == 'Food101':
        from models.Food101Models import Food101Model as VisionModel
        model_args = ()
    else:
        raise ValueError()
    return VisionModel, model_args

def getImageLoader(dataset, setting):
    if setting == "realistic":
        def load_realistic(path):
            image = Image.open(path).convert("RGB")
            return np.asarray(image, dtype=np.float32)

        return load_realistic

    if dataset == "Flower102":
        resize = transforms.Resize((256, 256))
    elif dataset == "Food101":
        resize = transforms.Resize((224, 224))
    else:
        resize = transforms.Resize(256)
    operations = [resize]
    if dataset != "Food101":
        operations.append(transforms.CenterCrop(224))
    operations.append(transforms.ToTensor())
    transform = transforms.Compose(operations)

    def load_common(path):
        image = Image.open(path).convert("RGB")
        return pytorch_switch(transform(image)).detach().numpy()

    return load_common

def getLabelFile(args):
    if args.demo:
        if args.dataset != "ImageNet1K":
            raise ValueError("--demo is only supported with --dataset ImageNet1K")
        return "TEST_IMGs/demo_targeted.json" if args.attack_type == "targeted" else "TEST_IMGs/demo_untargeted.json"
    if args.dataset == "Flower102":
        return "TEST_IMGs/EfficientNetV2S_Flower102.json"
    if args.dataset == "Food101":
        return "TEST_IMGs/Swin_Food101.json"
    return f"TEST_IMGs/{args.vision_model}_ImgNet1K.json"
