import torch
from models import AlexNet, resnet18, resnet50


def get_model(device, model_name, weight_path=None):
    if model_name == 'alexnet':
        model = AlexNet().to(device)
    elif model_name == 'resnet18':
        model = resnet18().to(device)
    elif model_name == 'resnet50':
        model = resnet50().to(device)
    else:
        raise NotImplementedError
    if weight_path:
        if torch.cuda.is_available():
            weights = torch.load(weight_path)
        else:
            weights = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(weights)
    return model
