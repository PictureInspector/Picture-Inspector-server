import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Encoder(nn.Module):
    def __init__(self, embed_size: int,  dropout: int = 0.5, train_conv: bool = False) -> None:
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

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


class Decoder(nn.Module):
    def __init__(self, attention_dim: int, decoder_dim: int, embed_dim: int, voc_size: int, encoder_dim: int = 2048,
                 dropout: float = 0.5) -> None:
        """
        :param attention_dim: size of the attention network
        :param decoder_dim: size of decoder's RNN
        :param embed_dim: embedding size
        :param voc_size: size of the vocabulary
        :param encoder_dim: size of encoded images
        :param dropout: dropout probability
        """
        super(Decoder, self).__init__()
        self.voc_size = voc_size

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embed = nn.Embedding(voc_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.initial_hidden = nn.Linear(encoder_dim, decoder_dim)
        self.initial_cell = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # for a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, voc_size)

        # initialize weights of some layers with the uniform distribution
        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize weights of some layers with the uniform distribution
        :return: None
        """
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the hidden and cell states of decoder's LSTM based on the encoded images
        :param encoder_out: encoded images
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        hidden = self.initial_hidden(mean_encoder_out)
        cell = self.initial_cell(mean_encoder_out)
        return hidden, cell

    def forward(self, encoder_out: torch.Tensor, encoded_captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation of the decoder
        :param encoder_out: encoded images
        :param encoded_captions: encoded captions
        :return: scores for each word in the vocabulary, weights
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        decode_length = encoded_captions.size(0) - 1
        encoded_captions = encoded_captions.permute((1, 0))
        voc_size = self.voc_size

        # Flatten the image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        embeds = self.embed(encoded_captions)

        # Initialize the hidden and cell states of the LSTM
        hidden, cell = self.init_hidden_state(encoder_out)

        predictions = torch.zeros(batch_size, decode_length, voc_size).to(device)
        alphas = torch.zeros(batch_size, decode_length, num_pixels).to(device)

        for t in range(decode_length):
            # apply the attention to the captions
            alpha, attention_weighted_encoding = self.attention(encoder_out, hidden)
            # apply gate to the hidden state
            gate = self.sigmoid(self.f_beta(hidden))
            attention_weighted_encoding = gate * attention_weighted_encoding
            # get the next hidden and cell states of LSTM
            #print(encoded_captions.shape)
            #print(embeds.shape, attention_weighted_encoding.shape)
            hidden, cell = self.lstm_cell(torch.cat([embeds[:, t, :], attention_weighted_encoding], dim=1),
                                          (hidden, cell))
            # get the predictions for each word in the vocabulary
            preds = self.fc(self.dropout(hidden))
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha

        return predictions, alphas
