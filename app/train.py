import glob
from collections import Counter
from timeit import default_timer as timer
from typing import Iterable, List

import MeCab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import vocab

from app.models import Seq2SeqTransformer
from app.utils import generate_square_subsequent_mask, create_mask


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


class Symbol:
    def __init__(self):
        self.unk_idx = 0
        self.pad_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3
        self.specials = ['<unk>', '<pad>', '<bos>', '<eos>']


class Preprocessor:

    def __init__(self, symbol: Symbol):
        self.symbol = symbol
        self.tokenizer = MeCab.Tagger('-Owakati')
        self.vocab = None
        self.lines = []
        self.text_transforms = sequential_transforms(
            self.token_transform,
            self.vocab_transform,
            self.tensor_transform)

    def clean(self, line):
        return line.strip('\n').split('\t')

    def token_transform(self, line):
        """ 単語に分割した配列を返す処理"""
        parsed = self.tokenizer.parse(line).split(' ')
        return parsed

    def vocab_transform(self, tokens):
        """ 単語の配列を渡して、IDの配列を返す処理"""
        return [self.vocab[token] for token in tokens]

    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.symbol.bos_idx]),
                          torch.tensor(token_ids),
                          torch.tensor([self.symbol.eos_idx])))

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transforms(src_sample.rstrip('\n')))
            tgt_batch.append(self.text_transforms(tgt_sample.rstrip('\n')))

        src_batch = pad_sequence(src_batch, padding_value=self.symbol.pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.symbol.pad_idx)
        return src_batch, tgt_batch

    def create_vocab(self, dir_path):
        counter = Counter()
        for path in glob.glob(f'{dir_path}/*'):
            with open(path, 'rt') as fin:
                for line in fin:
                    q, a = self.clean(line)
                    counter.update(self.token_transform(line))
                    counter.update(self.token_transform(line))
                    self.lines.append((q, a))
        self.vocab = vocab(counter)
        [self.vocab.insert_token(token, i) for (i, token) in enumerate(self.symbol.specials)]
        self.vocab.set_default_index(self.symbol.unk_idx)


class Trainer:

    def __init__(self, vocabulary, symbol: Symbol, text_transforms, collate_fn):
        self.text_transforms = text_transforms
        self.collate_fn = collate_fn
        self.symbol = symbol
        self.vocab = vocabulary

        # Hyper parameters
        self.emb_size = 512
        self.nhead = 8
        self.ffn_hid_dim = 512
        self.batch_size = 128
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3

        self.source_vocab_size = len(self.vocab)
        self.target_vocab_size = len(self.vocab)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = Seq2SeqTransformer(self.num_encoder_layers,
                                   self.num_decoder_layers,
                                   self.emb_size,
                                   self.nhead,
                                   self.source_vocab_size,
                                   self.target_vocab_size,
                                   self.ffn_hid_dim)

        # xavierの初期値
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.model = model.to(self.device)

        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.symbol.pad_idx)

        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=0.0001,
                                          betas=(0.9, 0.98),
                                          eps=1e-9)

    def train(self, sentences):
        self.model.train()
        losses = 0
        train_iter = sentences
        train_dataloader = DataLoader(train_iter, batch_size=self.batch_size, collate_fn=self.collate_fn)

        for src, tgt in train_dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            self.optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = self.loss_func(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            self.optimizer.step()
            losses += loss.item()

        return losses / len(train_dataloader)
        pass

    def evaluate(self, sentences):
        self.model.eval()
        losses = 0

        val_iter = sentences
        val_dataloader = DataLoader(val_iter, batch_size=self.batch_size, collate_fn=self.collate_fn)

        for src, tgt in val_dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = self.loss_func(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(val_dataloader)

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):
            memory = memory.to(self.device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(self.device)
            out = self.model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.symbol.eos_idx:
                break
        return ys

    def predict(self, src_sentence):
        self.model.eval()
        src = self.text_transforms(src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(
            self.model, src, src_mask, max_len=num_tokens+5, start_symbol=self.symbol.bos_idx).flatten()
        return " ".join(self.vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


def main():
    symbol = Symbol()
    preprocessor = Preprocessor(symbol)
    # 文を処理して語彙の辞書をつくる
    preprocessor.create_vocab('../data/narou/parsed')
    # 語彙の辞書を渡して訓練器をつくる
    trainer = Trainer(preprocessor.vocab,
                      symbol,
                      preprocessor.text_transforms,
                      preprocessor.collate_fn)

    num_epochs = 50

    for epoch in range(1, num_epochs+1):
        start_time = timer()
        train_loss = trainer.train(preprocessor.lines)
        end_time = timer()
        val_loss = trainer.evaluate(preprocessor.lines)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")


if __name__ == '__main__':
    main()