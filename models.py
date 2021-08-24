import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Encoder(nn.Module):
    def __init__(self, encoded_size: int = 14, train_conv: bool = False) -> None:
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

    def forward(self, imgs: torch.tensor) -> torch.tensor:
        """
        Forward propagation of Encoder
        :param imgs: a batch of images
        :return: Tensor with encoded images of shape (batch_size, encoded_size, encoded_size, 2048)
        """
        out = self.resnet(imgs)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out


class Attention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int) -> None:
        """
        Initialize the Attention class
        :param encoder_dim: size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim)
        self.full_attention = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.tensor, decoder_hidden: torch.tensor) -> tuple:
        """
        Forward propagation of Attention
        :param encoder_out: encoded images
        :param decoder_hidden: previous encoded output
        :return: weights, attention encoding
        """
        att1 = self.encoder_attention(encoder_out)
        att2 = self.decoder_attention(decoder_hidden)
        att = self.full_attention(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return alpha, attention_encoding

