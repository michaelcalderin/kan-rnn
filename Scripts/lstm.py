import argparse
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, random_split


# text helpers
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class Vocab:
    PAD = 0
    UNK = 1

    def __init__(self):
        self.token2idx = {"<pad>": self.PAD, "<unk>": self.UNK}
        self.idx2token = {self.PAD: "<pad>", self.UNK: "<unk>"}

    def add(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def encode(self, token):
        return self.token2idx.get(token, self.UNK)

    def __len__(self):
        return len(self.token2idx)


def build_vocab(sentences):
    vocab = Vocab()
    for sent in sentences:
        for tok in tokenize(sent):
            vocab.add(tok)
    return vocab


# GloVe
def load_glove(path):
    """Return {word: np.array} for every line in the GloVe text file."""
    vectors = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            vectors[word] = vec
    return vectors


def build_embedding_matrix(vocab, glove, embed_dim):
    matrix = np.random.normal(0, 0.1, (len(vocab), embed_dim)).astype(np.float32)
    matrix[Vocab.PAD] = 0.0
    hit = 0
    for token, idx in vocab.token2idx.items():
        if token in glove:
            matrix[idx] = glove[token]
            hit += 1
    coverage = hit / max(len(vocab) - 2, 1) * 100
    print(f"GloVe coverage: {hit}/{len(vocab)-2} vocab tokens ({coverage:.1f}%)")
    return matrix


# dataset
LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2}


class SentimentDataset(Dataset):
    def __init__(self, sentences, labels, vocab, max_len=128):
        self.vocab = vocab
        self.max_len = max_len
        self.samples = []
        for sent, label in zip(sentences, labels):
            tokens = tokenize(sent)[: max_len]
            ids = [vocab.encode(t) for t in tokens]
            if len(ids) > 0:
                self.samples.append((ids, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    ids_list, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids_list], dtype=torch.long)
    max_len = lengths[0].item()
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


# model
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_classes, dropout, pretrained_weights):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=Vocab.PAD)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))

        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        emb = self.drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=True)
        _, (hidden, _) = self.rnn(packed)
        out = hidden[-1]
        return self.fc(self.drop(out))


# training loop
def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            total += y.size(0)
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / total, f1


# main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.csv")
    p.add_argument("--glove_path", default="glove.6B.100d.txt")
    p.add_argument("--embed_dim", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--freeze_embeddings", action="store_true")
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--lr_patience", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # data loading
    df = pd.read_csv(args.data)

    # handle slight column name variation
    sentence_col = "Sentence"
    label_col = "Sentiment"
    df[label_col] = df[label_col].str.lower().str.strip()
    df = df[df[label_col].isin(LABEL_MAP)].reset_index(drop=True)
    
    # remove duplicate sentences
    df = df.drop_duplicates(subset=sentence_col, keep=False).reset_index(drop=True)
    sentences = df[sentence_col].tolist()
    labels = df[label_col].map(LABEL_MAP).tolist()
    print(f"samples: {len(sentences)}  | label dist: {df[label_col].value_counts().to_dict()}")

    # split into train/val/test
    n_total = len(sentences)
    n_test = max(1, int(n_total * 0.1))
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val - n_test
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    train_sentences = [sentences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_sentences = [sentences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_sentences = [sentences[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    # build vocab and embeddings ONLY on training data
    vocab = build_vocab(train_sentences)
    print(f"vocab size: {len(vocab)}")
    glove = load_glove(args.glove_path)
    embed_matrix = build_embedding_matrix(vocab, glove, args.embed_dim)
    del glove

    # datasets
    train_set = SentimentDataset(train_sentences, train_labels, vocab, max_len=args.max_len)
    val_set = SentimentDataset(val_sentences, val_labels, vocab, max_len=args.max_len)
    test_set = SentimentDataset(test_sentences, test_labels, vocab, max_len=args.max_len)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # model
    num_classes = len(LABEL_MAP)
    model = SentimentRNN(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
        pretrained_weights=embed_matrix,
    ).to(device)
    if args.freeze_embeddings:
        model.embedding.weight.requires_grad_(False)

    # class weights from training set only
    train_label_counts = pd.Series(train_labels).value_counts()
    class_weights = torch.tensor(
        [len(train_labels) / (num_classes * train_label_counts.get(k, 1)) for k in LABEL_MAP],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )


    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience
    )

    best_val_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_f1={train_f1:.4f}  "
            f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  "
            f"({elapsed:.1f}s)"
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_f1)

        # print when LR changes
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != prev_lr:
            print(f"Learning rate changed: {prev_lr:.6f} -> {new_lr:.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "lstm_model.pt")

    print(f"best val_f1: {best_val_f1:.4f}  -> weights saved to best_model.pt")

    # evaluate on test set ONCE after training
    test_loss, test_f1 = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"Test set: loss={test_loss:.4f}  f1={test_f1:.4f}")

    # plotting
    import matplotlib.pyplot as plt
    epochs = range(1, args.epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 1.25)
    plt.legend()
    plt.title('Loss Curves')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train F1')
    plt.plot(epochs, history['val_acc'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('F1 Score Curves')
    plt.tight_layout()
    plt.savefig('lstm_training_curves.png')
    plt.close()

    # confusion matrix on test set
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, lengths, y in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            logits = model(x, lengths)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(LABEL_MAP.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Test Confusion Matrix')
    plt.savefig('lstm_confusion_matrix.png')
    plt.close()


if __name__ == "__main__":
    main()
