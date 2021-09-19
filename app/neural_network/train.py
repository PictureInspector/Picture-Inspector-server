import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from app.neural_network.dataset import FlickrDataset, MyCollate
from app.neural_network.models import Model
from app.neural_network.utils import load_checkpoint, save_checkpoint
from app.neural_network.utils import caption_samples


pad_token = "<PAD>"

# Data parameters
img_folder = "./app/neural_network/data/images"  # folder with images
captions_file = "./app/neural_network/data/captions.txt"  # file with captions

# Training parameters
batch_size = 32
workers = 2
lr = 1e-3  # learning rate
checkpoint_path = "checkpoint.pth"  # path from which entity and optimizer are loaded and where they are saved
model_path = "entity.pth"  # path to which entity is saved
dataset_path = "dataset.pth"
load_model = False  # whether to load entity or not
save_model = True  # whether to save entity or not
step = 0  # starting epoch
epochs = 10  # the total number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for entity and PyTorch tensors
print_freq = 100  # how frequently print the information during the training
cudnn.benchmark = True

# Model parameters
embed_dim = 256  # embedding size
decoder_dim = 256  # dimension of decoder RNN
dropout = 0.5
train_conv = False  # whether to train ResNet or not
num_layers = 1  # the number of layers in LSTM


def main() -> None:
    """
    Main function for the training
    :return: None
    """
    global step
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # create the dataset
    dataset = FlickrDataset(img_folder, captions_file, transform)
    torch.save(dataset, "dataset.pth")

    # Split the dataset with 20% to be in the validation dataset
    test_size = len(dataset) // 5
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])

    # get the index for padding
    pad_idx = dataset.voc.wrd2idx[pad_token]

    # get dataloaders from the datasets
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=workers, shuffle=True,
                              pin_memory=True, collate_fn=MyCollate(pad_idx=pad_idx))
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=workers, shuffle=True,
                            pin_memory=True, collate_fn=MyCollate(pad_idx=pad_idx))

    # Initialize the entity, criterion and optimizer
    model = Model(embed_dim, decoder_dim, len(dataset.voc), num_layers, dropout, train_conv).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.voc.wrd2idx[pad_token])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # if load_model is True, load the entity and optimizer from the checkpoint
    if load_model:
        checkpoint = torch.load(checkpoint_path)
        step = load_checkpoint(checkpoint, model, optimizer)

    for epoch in range(step, epochs):
        # save_model is True, save the entity to the checkpoint
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": epoch,
            }
            save_checkpoint(checkpoint, checkpoint_path)
            torch.save(model, "entity.pth")
        # Training iteration
        train(model, train_loader, criterion, optimizer)
        # Validation iteration
        validate(model, val_loader, criterion)
        # Print some examples
        caption_samples("examples", model, dataset.voc, device, transform)


def train(model: Model, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Optimizer) -> None:
    """
    Iteration of the training process
    :param model: entity to be trained
    :param train_loader: training dataloader
    :param criterion: criterion for loss computation
    :param optimizer: optimizer of the entity
    :return: None
    """
    # set the entity to the training mode
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


def validate(model: Model, val_loader: DataLoader, criterion: nn.CrossEntropyLoss) -> None:
    """
    Validation of the entity
    :param model: entity to be validated
    :param val_loader: dataloader for validation
    :param criterion: criterion for loss computation
    :return: None
    """
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
