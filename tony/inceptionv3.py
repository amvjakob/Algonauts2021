import torch

__all__ = ['inceptionv3']

def inceptionv3(pretrained=True, **kwargs):
    model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', 
      pretrained=pretrained,**kwargs)

    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    return model

