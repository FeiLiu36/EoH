import torchvision
import timm
import torch
import torch.nn as nn
import os
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = "{}/../models".format(ABS_PATH)


def get_model(model_name="cifar10", model_path=None):
    assert model_name in ["cifar10", "imagenet"]
    assert model_path is not None
    return {
        "cifar10": get_cifar10_model,
        "imagenet": get_imagenet_model
    }[model_name](model_path=model_path)


def get_cifar10_model(model_path=None):
    preprocessor = torchvision.transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.201]
    )

    net = timm.create_model("resnet18", pretrained=False)
    # override model
    net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    net.maxpool = nn.Identity()  # type: ignore
    net.fc = nn.Linear(512, 10)

    net.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
            model_dir=model_path,
            map_location="cpu",
            file_name="resnet18_cifar10.pth",
        )
    )
    model = nn.Sequential(
        preprocessor,
        net
    )
    model.eval()
    return model


def get_imagenet_model(model_path=None):
    preprocessor = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    net = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    )
    model = nn.Sequential(
        preprocessor,
        net
    )
    model.eval()
    return model
