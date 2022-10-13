import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

UNKNOWN_BIGRAM = '<UNK>'
LEARNING_RATE = 1e-3
BATCH_SIZE = 20
NUMBER_OF_EPOCHS = 100


class LangDataset(Dataset):
    """
    Pytorch dataset class that accepts a text path, and optionally label path and
    a vocabulary. This class holds all the data, and implements
    a __getitem__ method to be used by a Python generator object or other classes that need it.
    """

    def __init__(self, text_path, label_path=None, trained_vocab=None):
        """
        Reads the content of text_file, optionally label_file and vocabulary

        Parameters
        ----------
        text_path : str
            Path to the text file
        label_path : str
            Path to the label file (default is None)
        trained_vocab : dict
            Vocabulary of the dataset (default is None)
        """
        with open(text_path, 'r+') as text_file:
            self.sentences = [_.strip() for _ in text_file.readlines()]

        if label_path is not None:
            with open(label_path, 'r+') as label_file:
                self.labels = [_.strip() for _ in label_file.readlines()]

        if trained_vocab is None:
            i = 0
            self.vocabulary = dict()
            for sentence in self.sentences:
                bigrams = [sentence[k] + sentence[k + 1] for k in range(len(sentence) - 1)]
                for bigram in bigrams:
                    if bigram not in self.vocabulary:
                        self.vocabulary[bigram] = i
                        i += 1

            self.vocabulary[UNKNOWN_BIGRAM] = i

            j = 0
            self.classes = dict()
            for label in self.labels:
                if label not in self.classes:
                    self.classes[label] = j
                    j += 1

            self.is_training = True
        else:
            self.vocabulary = trained_vocab['text']
            self.classes = trained_vocab['labels']

            self.is_training = False

    def vocab_size(self):
        """
        Informs the vocab size

        Returns
        -------
        int
            Size of the vocabulary
        int
            Number of class labels
        """
        return len(self.vocabulary), len(self.classes)

    def __len__(self):
        """
        Informs the number of instances in the data

        Returns
        -------
        int
            Number of instances
        """
        return len(self.sentences)

    def __getitem__(self, i):
        """
        Returns the i-th instance in the format of (text, label)
        Text and label are encoded according to the vocab (word_id)

        Returns
        -------
        tensor
            Encoded text
        tensor
            Encoded label

        """
        sentence = self.sentences[i]
        bigrams = [sentence[k] + sentence[k + 1] for k in range(len(sentence) - 1)]

        text = list()
        for b in bigrams:
            if b in self.vocabulary:
                text.append(self.vocabulary[b])
            else:
                text.append(self.vocabulary[UNKNOWN_BIGRAM])

        text = torch.LongTensor(text)

        if self.is_training:
            label = self.classes[self.labels[i]]
        else:
            # default label of each text, not going to be used when testing
            label = torch.LongTensor([-1])

        return text, label


class Model(nn.Module):
    """
    Model class with one Embedding layer with dimension 16,
    a feed-forward layer that reduces the dimension from 16 to 200 with ReLU activation,
    a dropout layer, and a feed-forward layer that reduces the dimension from 200 to number of classes of labels.
    """

    def __init__(self, num_vocab, num_class, dropout=0.3):
        """
        Creates a model instance

        Parameters
        ----------
        num_vocab : int
            Number of distinct words in the vocabulary
        num_class : int
            Number of classes of labels in the dataset
        dropout : double
            Fraction of dropout (default is 0.3)
        """
        super(Model, self).__init__()
        self.embedding_layer = nn.Embedding(num_vocab, 16)

        self.fc1 = nn.Linear(16, 200)
        self.fc2 = nn.Linear(200, num_class)

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        """
        Executes the forward pass of training

        Parameters
        ----------
        x : tensor
            Input instance for the forward pass

        Returns
        -------
        tensor
            Output of the forward pass

        """
        model = self.embedding_layer(x)
        model = model.mean(dim=1)

        model = self.fc1(model)
        model = self.relu(model)

        model = self.dropout_layer(model)
        model = self.fc2(model)

        return model


def collator(batch):
    """
    Collates the batch of input instances by transforming them into uniform dimension tensors with required
    padding amount

    Parameters
    ----------
    batch : list
        List of (text, label) pairs

    Returns
    -------
    tensor
        Sequence of padded text instances
    tensor
        Encoded label as a tensor
    """
    texts = [b[0] for b in batch]
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)

    labels = torch.tensor([b[1] for b in batch])

    return texts, labels


def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None):
    """
    Defines the training procedure according to the specified parameters

    Parameters
    ----------
    model : Model
        Model instance
    dataset : LangDataset
        Language dataset
    batch_size : int
        Size of an individual mini batch
    learning_rate : double
        Learning rate of the algorithm
    num_epoch : int
        Number of training epochs
    device : str
        Device to execute the training procedure on (default is cpu)
    model_path : str
        Path specifying where to save the model (default is None)
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            texts = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            scores = model(texts)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 100 == 99:
                print(f'[{epoch + 1}, {step + 1}] loss: {running_loss / 100}')
                running_loss = 0.0

    end = datetime.datetime.now()

    checkpoint = {
        'state_dict': model.state_dict(),
        'trained_vocab': {
            'text': dataset.vocabulary,
            'labels': dataset.classes
        }
    }

    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print(f'Training finished in {(end - start).seconds / 60.0} minutes.')


def test(model, dataset, class_map, device='cpu'):
    """
    Defines the testing procedure for the model

    Parameters
    ----------
    model : Model
        Model to be used in the testing procedure
    dataset : LangDataset
        Testing dataset with new instances and default labels
    class_map : dict
        Dictionary defining the mapping between index and the class id
    device : str
        Device to run the testing procedure on

    Returns
    -------
    list
        List of labels predicted
    """
    model.eval()

    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    labels = []
    with torch.no_grad():
        for data in data_loader:
            texts = data[0].to(device)
            outputs = model(texts).cpu()

            _, res = outputs.max(1)
            labels += res

    labels = [class_map[label.item()] for label in labels]

    return labels


def main(args):
    if torch.cuda.is_available():
        device_str = f'cuda:{0}'
    else:
        device_str = 'cpu'
    device = torch.device(device_str)

    assert args.train or args.test, "Please specify --train or --test"

    if args.train:
        assert args.label_path is not None, "Please provide the labels for training using --label_path argument"

        dataset = LangDataset(args.text_path, args.label_path)

        num_vocab, num_class = dataset.vocab_size()
        model = Model(num_vocab, num_class).to(device)

        train(model, dataset, BATCH_SIZE, LEARNING_RATE, NUMBER_OF_EPOCHS, device_str, args.model_path)

    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"

        checkpoint = torch.load(args.model_path)
        dataset = LangDataset(args.text_path, trained_vocab=checkpoint['trained_vocab'])

        num_vocab, num_class = len(checkpoint['trained_vocab']['text']), len(checkpoint['trained_vocab']['labels'])

        model = Model(num_vocab, num_class).to(device)
        model.load_state_dict(checkpoint['state_dict'])

        lang_map = checkpoint['trained_vocab']['labels']

        assert isinstance(lang_map, dict)
        lang_map = dict(map(reversed, lang_map.items()))

        predictions = test(model, dataset, lang_map, device_str)

        with open(args.output_path, 'w+', encoding='utf-8') as out:
            out.write('\n'.join(predictions))

    print('\n==== A2 Part 2 Done ====')


def get_arguments():
    """
    Parses arguments from the command line interface

    Returns
    -------
    Namespace
        Arguments parsed

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', help='path to the text file')
    parser.add_argument('--label_path', default=None, help='path to the label file')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--model_path', required=True, help='path to the output file during testing')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
