import torch
import torch.nn as nn
import torchvision


class Encoder(nn.Module):
    def __init__(self, embed_size: int,  dropout: float = 0.5, train_conv: bool = False) -> None:
        """
        Initialize the encoder class
        :param embed_size: size of the result tensor
        :param dropout: dropout probability
        :param train_conv: whether to train ResNet or not
        """
        super(Encoder, self).__init__()

        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Set ResNet gradient according to train_conv
        for name, param in self.resnet.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = train_conv

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of Encoder
        :param imgs: a batch of images
        :return: Tensor with encoded images of shape (batch_size, embed_size)
        """
        out = self.resnet(imgs)
        return self.dropout(self.relu(out))


class Decoder(nn.Module):
    def __init__(self, decoder_dim: int, embed_dim: int, voc_size: int, num_layers: int, dropout: float = 0.5) -> None:
        """
        :param decoder_dim: size of decoder's RNN
        :param embed_dim: embedding size
        :param voc_size: size of the vocabulary
        :param num_layers: the number of layers in LSTM
        :param dropout: dropout probability
        """
        super(Decoder, self).__init__()

        self.embed = nn.Embedding(voc_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(embed_dim, decoder_dim, num_layers)
        self.fc = nn.Linear(decoder_dim, voc_size)

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of the decoder
        :param encoder_out: encoded images
        :param captions: encoded captions
        :return: scores for each word in the vocabulary
        """
        embeds = self.dropout(self.embed(captions))
        embeds = torch.cat((encoder_out.unsqueeze(0), embeds), dim=0)
        hidden, _ = self.lstm(embeds)
        out = self.fc(hidden)

        return out


class Model(nn.Module):
    def __init__(self, embed_dim: int, decoder_dim: int, voc_size: int, num_layers: int, dropout: float = 0.5,
                 train_conv: bool = False) -> None:
        """
        Initialization of the model
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param voc_size: size of the vocabulary
        :param num_layers: the number of layers in LSTM
        :param dropout: dropout probability
        :param train_conv: whether to train ResNet or not
        """
        super(Model, self).__init__()
        self.encoder = Encoder(embed_dim, dropout, train_conv)
        self.decoder = Decoder(decoder_dim, embed_dim, voc_size, num_layers)

    def forward(self, imgs: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of the model
        :param imgs: a batch of images
        :param captions: encoded captions
        :return: scores for each word in the vocabulary
        """
        encoder_out = self.encoder(imgs)
        out = self.decoder(encoder_out, captions)

        return out
