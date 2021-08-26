import torch
import torchvision.transforms as transforms
from dataset import get_dataloader
from models import Encoder, Decoder

# Data parameters
img_folder = "./data/images"  # folder with images
captions_file = "./data/captions.txt"  # file with captions

embed_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

cur_epoch = 0
epochs = 20
epochs_since_improvement = 0
batch_size = 32
workers = 1
encoder_lr = 1e-2
decoder_lr = 1e-2
grad_clip = 5.
alpha_c = 1.
best_bleu4 = 0.
print_freq = 100
train_encoder = False
checkpoint = None


def main():
    global best_bleu4, epochs_since_improvement, checkpoint, cur_epoch, train_encoder, img_folder, captions_file
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset, dataloader = get_dataloader(img_folder, captions_file, transform, batch_size, shuffle=True, pin_memory=True)
    if checkpoint is None:
        decoder = Decoder(attention_dim, decoder_dim, embed_dim, len(dataset.voc), dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder(train_conv=train_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if train_encoder else None
    else:
        checkpoint = torch.load(checkpoint)
        cur_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['best_bleu4']

    encoder = encoder.to(device)
    decoder = decoder.to(device)
