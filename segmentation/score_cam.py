from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from data import get_img, H, W
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_binary_cam(model, model_name, img_path):
    img = get_img(img_path, device)
    score_cam = ScoreCam(model, model_name)
    cam = score_cam.generate_cam(img, 1)
    binary_cam = np.zeros(cam.shape, dtype=bool)
    binary_cam[cam>0.5] = True
    return binary_cam


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        # self.target_layer = target_layer

    def forward_pass(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        if self.model_name == 'alexnet':
            for module_pos, module in self.model.features._modules.items():
                x = module(x)  # Forward
                if int(module_pos) == 11:
                    conv_output = x  # Save the convolution output on that layer
            x = x.view(x.size(0), -1)  # Flatten
            # Forward pass on the classifier
            x = self.model.classifier(x)
        elif self.model_name.find('resnet') != -1:
            for module_pos, module in self.model._modules.items():
                x = module(x)  # Forward
                if module_pos == 'layer4':
                    conv_output = x  # Save the convolution output on that layer
                if module_pos == 'avgpool':
                    x = x.view(x.size(0), -1)
        else:
            raise NotImplementedError
        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """

    def __init__(self, model, model_name):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, model_name)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.cpu().numpy())
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :], 0), 0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(H, W), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            w = F.softmax(self.extractor.forward_pass(input_image * norm_saliency_map)[1], dim=1)[0][target_class]
            cam += w.data.cpu().numpy() * target[i, :, :].data.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[3],
                                                    input_image.shape[2]), Image.ANTIALIAS)) / 255
        return cam
