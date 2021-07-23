# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import itertools

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchtext import data, datasets

import argparse

parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--output_dir", nargs='?', type=str, help="output directry")
parser.add_argument("--model", action="store_true", help="trained model")
parser.add_argument("--train", nargs='?', type=str, help="training data")
parser.add_argument("--test", nargs='?', type=str, help="test data")
parser.add_argument("--vis", nargs='?', type=str, default=None, help="visualize attention")
parser.add_argument("--format", nargs='?', type=str, default="fol", help="formula format")
parser.add_argument("--epochs", nargs='?', type=int, default=10, help="number of epochs")
parser.add_argument("--maxlength", nargs='?', type=int, default=60, help="max length")
parser.add_argument("--size", nargs='?', type=int, default=100000000, help="training set size")
parser.add_argument("--hidden_size", nargs='?', type=int, default=256, help="hidden size")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = args.maxlength

def tokenize_sentence(sentence):
    return [word for word in sentence.split(' ')]

def tokenize_formula(formula):
    return [atom for atom in split_nltkformula(formula)]

def tokenize_other(formula):
    return [atom for atom in formula.split(' ')]

def split_prologformula(formula):
    list_formula = []
    tmp = []
    for f in formula:
      if f == "_":
        list_formula.append(f)
      elif f == "(" or f == ")" or f == ",":
          if len(tmp) >= 1:
              list_formula.append("".join(tmp))
              tmp = []
          list_formula.append(f)
      else:
          tmp.extend(f)
    return list_formula

def split_nltkformula(formula):
    list_formula = []
    tmp = []
    for i, f in enumerate(formula):
        if f == "-":
            if formula[i+1] == ">":
                tmp.extend(f)
                continue
            #negation
            list_formula.append(f)
        elif f == "(" or f == ")" or f == "," or f == "." or f == ">":
            if len(tmp) >= 1 and f != ">":
                list_formula.append("".join(tmp))
                tmp = []
            elif len(tmp) >=1 and f == ">":
                tmp.extend(f)
                continue
            list_formula.append(f)
        elif f == " ":
            if formula[i+1] == "&" or formula[i+1] == "|" or formula[i+2] == ">":
                tmp.extend(f)
                continue
            else:
                tmp.extend(f)
                list_formula.append("".join(tmp))
                tmp = []
        else:
            tmp.extend(f)
    return list_formula

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = MAX_LENGTH):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])        
        self.dropout = nn.Dropout(dropout)        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):               
        batch_size = src.shape[0]
        src_len = src.shape[1]        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)         
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))            
        for layer in self.layers:
            src = layer(src, src_mask)                       
        return src

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)     
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim = -1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = MAX_LENGTH):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

    
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output, _ = model(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def trainIters(output_dir, model, n_iters, train_iterator, valid_iterator, learning_rate=0.0005):
    early = 0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_valid_loss = float('inf')

    for epoch in range(n_iters):
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, 1)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), output_dir+'/seq2fol_model.pt') 

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output, _ = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def list_split(xlist):
    res = []
    for x in xlist:
        res.extend(split_nltkformula(x))
    return res

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = MAX_LENGTH):
    model.eval()
    if isinstance(sentence, str):
        tokens = tokenize_sentence(sentence)
    else:
        tokens = sentence
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention

def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    assert n_rows * n_cols == n_heads
    fig = plt.figure(figsize=(15,25))
    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()
        cax = ax.matshow(_attention, cmap='bone')
        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    plt.close()

hidden_size = args.hidden_size
teacher_forcing_ratio = 0.5
batch_size = 128

src = data.Field(tokenize = tokenize_sentence, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = False, 
            batch_first = True)

trg, train_data_all = "", ""
if args.format == "fol":
    trg = data.Field(tokenize = tokenize_formula, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = False, 
                batch_first = True)
elif args.format == "clf" or args.format == "free":
    trg = data.Field(tokenize = tokenize_other, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = False, 
            batch_first = True)

if args.size == 100000000:
    train_data_all = data.TabularDataset(
            path=args.train,format='tsv',
            fields={'sentence': ('src', src),
                    'fol': ('trg', trg)
                    })
else:
    df = pd.read_csv(args.train, sep="\t")
    newdf = df[0:args.size]
    newdf.to_csv(args.train+"_"+str(args.size)+".tsv", sep="\t", index=False)
    train_data_all = data.TabularDataset(
            path=args.train+"_"+str(args.size)+".tsv",format='tsv',
            fields={'sentence': ('src', src), 
                    'fol': ('trg', trg)
                    })  

src.build_vocab(train_data_all, min_freq = 1)
trg.build_vocab(train_data_all, min_freq = 1)

input_dim = len(src.vocab)
output_dim = len(trg.vocab)
print(input_dim, output_dim)

src_pad_idx = src.vocab.stoi[src.pad_token]
trg_pad_idx = trg.vocab.stoi[trg.pad_token]

train_data, valid_data = train_data_all.split(split_ratio=0.9)
print(len(train_data), len(valid_data))
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data), 
     batch_size = batch_size,
     device = device,
     sort=False)

encoder1 = Encoder(input_dim, hidden_size, 3, 8, 512, 0.1, device)
decoder1 = Decoder(output_dim, hidden_size, 3, 8, 512, 0.1, device)
model = Seq2Seq(encoder1, decoder1, src_pad_idx, trg_pad_idx, device).to(device)
model.apply(initialize_weights);
# parameter
params = 0
for p in model.parameters():
    if p.requires_grad:
        params += p.numel()
print("number of parameter")       
print(params)

if args.model:
    model.load_state_dict(torch.load(args.output_dir+'/seq2fol_model.pt'))
else:
    trainIters(args.output_dir, model, args.epochs, train_iterator, valid_iterator)
    val_preds, val_golds = [], []
    for pair in valid_data.examples:
        output_words1, attentions1 = translate_sentence(pair.src,src,trg,model,device)
        if args.format == "fol":
            pred = "".join(output_words1[:-1])
            val_preds.append(pred)
            val_golds.append("".join(pair.trg))
        elif args.format == "clf" or args.format == "free":
            pred = " ".join(output_words1[:-1])
            val_preds.append(pred)
            val_golds.append(" ".join(pair.trg))
    val_parfect_acc = accuracy_score(val_golds, val_preds)
    print(f'validation parfect accuracy: {val_parfect_acc*100:.1f}')
    with open(args.output_dir+'/val_acc.txt', 'a') as g:
        g.write(f'{val_parfect_acc*100:.1f}\n')

eval_pairs = []
pred_pairs = []
golds = []
preds = []
with open(args.test, "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        line = line.strip()
        line_list = line.split("\t")
        if args.format == "clf":
            golds.append(line_list[4])
            eval_pairs.append([line_list[0], line_list[1], line_list[2], line_list[4], line_list[3]])
        else:
            golds.append(line_list[3])
            #id depth sentence sentence_fol phenomena_tags
            eval_pairs.append(line_list)

for pair in eval_pairs:
    output_words1, attentions1 = translate_sentence(pair[2],src,trg,model,device)
    if args.format == "fol":
        pred = "".join(output_words1[:-1])
        preds.append(pred)
        pred_pairs.append([pair[0], pair[1], pair[2], pair[3], pred, pair[4]])
    elif args.format == "clf" or args.format == "free":
        pred = " ".join(output_words1[:-1])
        preds.append(pred)
        pred_pairs.append([pair[0], pair[1], pair[2], pair[3], pred, pair[4]])


parfect_acc = accuracy_score(golds, preds)
print(f'test parfect accuracy: {parfect_acc*100:.1f}')
with open(args.output_dir+'/test_acc.txt', 'a') as g:
    g.write(f'{parfect_acc*100:.1f}\n')
predict_df = pd.DataFrame(pred_pairs, columns=["id", "depth", "sentence", "sentence_fol_gold", "sentence_fol_pred", "phenomena_tags"])
with open(args.output_dir+'/prediction.tsv', 'w') as f:
  f.write(predict_df.to_csv(sep="\t", index=False))
