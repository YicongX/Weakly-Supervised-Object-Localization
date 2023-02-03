import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),  #b*127*127*64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),  #b*63*63*64
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),  #b*63*63*192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),  #b*31*31*192
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #b*31*31*384
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  #b*31*31*256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  #b*31*31*256
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  #b*31*31*256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  #b*31*31*256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  #b*31*31*20
        )

    def forward(self, x):
        
        x = self.features(x)
        x = self.classifier(x)
        return x


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),  #b*127*127*64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),  #b*63*63*64
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),  #b*63*63*192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),  #b*31*31*192
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), #b*31*31*384
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  #b*31*31*256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  #b*31*31*256
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  #b*31*31*256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  #b*31*31*256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  #b*31*31*20
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Dropout2d(p=0.3,inplace = True),  
        )

        


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        #x1 = F.avg_pool2d(x, kernel_size = 3 , stride  = 1, padding = 1 )
        #x = (x + x1)/2

        return x


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    
    if pretrained == True:
        print("Load Pretrained")
        model.apply(weights_ini)
        alexnet = model_zoo.load_url(model_urls['alexnet'])
        model_dict = model.state_dict()
        alexnet_layers = {k: v for k, v in alexnet.items() if k.startswith('features')}
        model_dict.update(alexnet_layers)
        model.load_state_dict(model_dict)

    else:
        print("Train from scratch")
        model.apply(weights_ini)

    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    
    if pretrained == True:
        print("Load Pretrained")
        model.apply(weights_ini)
        alexnet = model_zoo.load_url(model_urls['alexnet'])
        model_dict = model.state_dict()
        alexnet_layers = {k: v for k, v in alexnet.items() if k.startswith('features')}
        model_dict.update(alexnet_layers)
        model.load_state_dict(model_dict)

    else:
        print("Train from scratch")
        model.apply(weights_ini)

    return model

def weights_ini(model):
    layer = model.__class__.__name__
    if layer.find('Conv') != -1:
        nn.init.xavier_normal_(model.weight.data)