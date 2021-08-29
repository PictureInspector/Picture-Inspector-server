import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from dataset import FlickrDataset, MyCollate
from models import Model
from utils import load_checkpoint, save_checkpoint
from tqdm import tqdm

pad_token = "<PAD>"

# Data parameters
img_folder = "./data/images"  # folder with images
captions_file = "./data/captions.txt"  # file with captions

# Training parameters
batch_size = 32
workers = 2
lr = 1e-3
checkpoint_path = "checkpoint.pth"
load_model = False
save_model = True
step = 0
epochs = 20

# Model parameters
embed_dim = 256  # embedding size
attention_dim = 256  # size of the attention network
decoder_dim = 256
dropout = 0.5
train_conv = False
num_layers = 1
print_freq = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
cudnn.benchmark = True


def main():
    global step
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
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=workers, shuffle=True,
                            pin_memory=True, collate_fn=MyCollate(pad_idx=pad_idx))
    model = Model(embed_dim, decoder_dim, len(dataset.voc), num_layers, dropout, train_conv).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.voc.wrd2idx[pad_token])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if load_model:
        checkpoint = torch.load(checkpoint_path)
        step = load_checkpoint(checkpoint, model, optimizer)

    model.train()

    for epoch in range(step, epochs):
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": epoch,
            }
            save_checkpoint(checkpoint, checkpoint_path)
            train(model, train_loader, criterion, optimizer)
            validate(model, val_loader, criterion)


def train(model: Model, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Optimizer) -> None:
    """
    Iteration of the training process
    :param model: model to be trained
    :param train_loader: training dataloader
    :param criterion: criterion for loss computation
    :param optimizer: optimizer of the model
    :return: None
    """
    # set the model to the training mode
    model.train()

    for idx, (imgs, captions) in enumerate(train_loader):
        # move data to GPU if available
        imgs = imgs.to(device)
        captions = captions.to(device)

        # Forward propagation
        out = model(imgs, captions[:-1])

        # Calculate loss
        loss = criterion(out.reshape(-1, out.shape[2]), captions.reshape(-1))

        # Back propagation
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Print training loss
        if idx % print_freq == 0:
            print(f"Training loss {idx}/{len(train_loader)}: {loss.item()}")


def validate(model: Model, val_loader: DataLoader, criterion: nn.CrossEntropyLoss):
    losses = []

    model.eval()

    for idx, (imgs, captions) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
        # move data to GPU if available
        imgs = imgs.to(device)
        captions = captions.to(device)

        # Forward propagation
        out = model(imgs, captions[:-1])

        # Calculate loss
        loss = criterion(out.reshape(-1, out.shape[2]), captions.reshape(-1))

        losses.append(loss.item())

    # Print training loss
    print(f"Validation loss: {np.mean(losses)}")


if __name__ == "__main__":
    main()
