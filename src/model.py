import torchvision.models as models
import torch.nn as nn

def get_model(model_name="resnet50", pretrained=True):
    """
    Returns a CNN model with the final layer adapted for binary classification.
    
    Args:
        model_name (str): Which architecture to use ('resnet50' or 'efficientnet_b0')
        pretrained (bool): Whether to load pretrained ImageNet weights
    
    Returns:
        model (nn.Module): PyTorch model
    """
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)

    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 2)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model
