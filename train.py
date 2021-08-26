import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from dataset import FlickrDataset, MyCollate
from models import Encoder, Decoder
from utils import adjust_lr
from tqdm import tqdm

pad_token = "<PAD>"

# Data parameters
img_folder = "./data/images"  # folder with images
captions_file = "./data/captions.txt"  # file with captions

# Model parameters
embed_dim = 512  # embedding size
attention_dim = 512  # size of the attention network
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

cur_epoch = 0
epochs = 30
epochs_since_improvement = 0
batch_size = 32
workers = 3
encoder_lr = 1e-2
decoder_lr = 1e-2
grad_clip = 5.
alpha_c = 1.
best_bleu4 = 0.
print_freq = 100
train_encoder = False
checkpoint = None


def main():
    global best_bleu4, epochs_since_improvement, checkpoint, cur_epoch, train_encoder, img_folder, captions_file, \
        workers

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = FlickrDataset(img_folder, captions_file, transform)
    test_size = len(dataset) // 5
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])
    pad_idx = dataset.voc.wrd2idx[pad_token]
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=workers, shuffle=True,
                              pin_memory=True, collate_fn=MyCollate(pad_idx=pad_idx))
    val_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=workers, shuffle=True,
                            pin_memory=True, collate_fn=MyCollate(pad_idx=pad_idx))
    if checkpoint is None:
        decoder = Decoder(attention_dim, decoder_dim, embed_dim, len(dataset.voc), dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(train_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if train_encoder else None
    else:
        checkpoint = torch.load(checkpoint)
        cur_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['best_bleu4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if train_encoder and encoder_optimizer is None:
            encoder.fine_tune(train_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(cur_epoch, epochs):
        if epochs_since_improvement == 4:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            adjust_lr(decoder_optimizer, 0.8)
        if train_encoder:
            adjust_lr(encoder_optimizer, 0.8)
        train(train_loader, encoder, decoder)


def train(train_loader, encoder, decoder):
    decoder.train()
    encoder.train()

    for i, (imgs, captions) in tqdm(enumerate(train_loader)):
        imgs = imgs.to(device)
        print(captions.shape)


if __name__ == "__main__":
    main()
