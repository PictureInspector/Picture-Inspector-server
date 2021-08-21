import torch
import spacy

spacy_eng = spacy.load("en")


class Vocabulary:
    def __init__(self, occur_threshold: int) -> None:
        """
        initialize the vocabulary class
        :param occur_threshold: threshold for a word to be added to the vocabulary
        """
        # index to word converter
        self.idx2wrd = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        # word to index converter
        self.wrd2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.occur_threshold = occur_threshold

    def __len__(self) -> int:
        """
        :return: the length of the vocabulary
        """
        return len(self.idx2wrd)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        tokenize the text using spacy tokenizer
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
        numericalize the text
        :param text: text to be numericalized
        :return: numericalized text
        """
        # tokenize the text
        tokenized_text = self.tokenize(text)

        # convert all tokens to indices
        return [
            self.wrd2idx[token] if token in self.wrd2idx else self.wrd2idx["<UNk>"]
            for token in tokenized_text
        ]
