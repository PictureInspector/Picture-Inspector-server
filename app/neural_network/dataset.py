import os
import spacy
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

spacy_eng = spacy.load('en_core_web_sm')
start_token = "<SOS>"
end_token = "<EOS>"
unk_token = "<UNK>"
pad_token = "<PAD>"


class Vocabulary:
    def __init__(self, occur_threshold: int) -> None:
        """
        initialize the vocabulary class
        :param occur_threshold: threshold for a word to be added to the vocabulary
        """
        # index to word converter
        self.idx2wrd = {0: pad_token, 1: start_token, 2: end_token, 3: unk_token}
        # word to index converter
        self.wrd2idx = {pad_token: 0, start_token: 1, end_token: 2, unk_token: 3}
        self.occur_threshold = occur_threshold

    def __len__(self) -> int:
        """
        :return: the length of the vocabulary
        """
        return len(self.idx2wrd)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        :param text: input text to be tokenized
        :return: tokenized text
        """
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list: list) -> None:
        """
        builds the vocabulary from the given sentences
        :param sentence_list: list of sentences the vocabulary should be built from
        :return: None
        """
        # dictionary of occurrences for each word
        occurs = {}
        # counter for the index
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                if word not in occurs:
                    occurs[word] = 1
                else:
                    occurs[word] += 1
                # if the word has the number of occurrences as the threshold, then add the word to the vocabulary
                if occurs[word] == self.occur_threshold:
                    self.wrd2idx[word] = idx
                    self.idx2wrd[idx] = word
                    idx += 1

    def numericalize(self, text: str) -> list[int]:
        """
        :param text: text to be numericalized
        :return: numericalized text
        """
        # tokenize the text
        tokenized_text = self.tokenize(text)

        # convert all tokens to indices
        return [
            self.wrd2idx[token] if token in self.wrd2idx else self.wrd2idx[unk_token]
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root: str, annotation_file: str, transform: transforms.Compose = None, occur_threshold: int = 5)\
            -> None:
        """
        Initialize the dataset
        :param root: directory with the images
        :param annotation_file: file with image filenames and their appropriate captions
        :param transform: transform to be applies to the image
        :param occur_threshold: threshold for a word to be added to the vocabulary
        """
        self.root = root
        self.df = pd.read_csv(annotation_file)
        self.transform = transform

        self.img_filenames = self.df["image"]
        self.captions = self.df["caption"]

        self.voc = Vocabulary(occur_threshold)
        # build vocabulary from the captions
        self.voc.build_vocabulary(self.captions.tolist())

    def __len__(self) -> int:
        """
        :return: length of the dataset
        """
        return len(self.df)

    def __getitem__(self, index: int) -> tuple:
        """
        :param index: index of the item
        :return: the image and the caption corresponding to that index
        """
        caption = self.captions[index]
        img_id = self.img_filenames[index]
        img = Image.open(os.path.join(self.root, img_id)).convert("RGB")

        # apply transforms to the image if there are any
        if self.transform is not None:
            img = self.transform(img)

        # add the start token and the end token to the numericalized text
        numericalized_caption = [self.voc.wrd2idx[start_token]]
        numericalized_caption += self.voc.numericalize(caption)
        numericalized_caption.append(self.voc.wrd2idx[end_token])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx: int) -> None:
        """
        Initialize Collate
        :param pad_idx: value by which captions will be padded
        """
        self.pad_idx = pad_idx

    def __call__(self, batch: list) -> tuple:
        """
        :param batch: sample of items from the dataset
        :return: images and padded captions
        """
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets
