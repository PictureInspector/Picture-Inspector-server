import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Encoder(nn.Module):
    def __init__(self, encoded_size:int = 14, train_conv: bool = False):
        """
        Initialize the encoder class
        :param encoded_size: size of the result image
        :param train_conv: whether to train ResNet or not
        """
        super(Encoder, self).__init__()
        self.enc_size = encoded_size

        resnet = torchvision.models.resnet50(pretrained=True)
        # remove linear and pool layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # resize image to a fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))

        for parameter in self.resnet.parameters():
            parameter.requires_grad = False

        # if we are fine-tuning ResNet, set requires_grad to True
        for conv in list(self.resnet.children())[5:]:
            for parameter in conv.parameters():
                parameter.requires_grad = train_conv

    def forward(self, imgs):
        """
        Forward propagation of Encoder
        :param imgs: a batch of images
        :return: Tensor with encoded images of shape (batch_size, encoded_size, encoded_size, 2048)
        """
        out = self.resnet(imgs)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out


model = Encoder().to(device)
