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

# Training parameters
batch_size = 32
workers = 2

# Model parameters
embed_dim = 256  # embedding size
attention_dim = 256  # size of the attention network
decoder_dim = 256
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main():
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


if __name__ == "__main__":
    main()
