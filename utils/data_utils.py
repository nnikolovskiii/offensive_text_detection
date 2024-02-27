import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm



class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens = self.df["tkns"][idx]
        label = self.df["Label"][idx]

        tokens_t = torch.tensor(tokens)
        label_t = torch.tensor(label)

        return tokens_t, label_t


def collate_batch(batch):
    avg_len = 0
    for i in range(len(batch)):
        sentence, label = batch[i]
        avg_len += len(sentence)

    avg_len = int(avg_len / len(batch))
    sentences = []
    labels = []
    for i in range(len(batch)):
        sentence, label = batch[i]
        labels.append(label)
        if len(sentence) >= avg_len:
            sentences.append(sentence[:avg_len])
        else:
            sentences.append(torch.cat((sentence, torch.tensor([0 for i in range(avg_len - len(sentence))]))))

    return torch.stack(sentences), torch.stack(labels)

def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0

    epoch_loss = 0

    for step, batch in enumerate(tqdm(data_loader, desc="Training...")):
      optimizer.zero_grad()

      token_ids,labels = batch

      token_ids = token_ids.to(device)
      labels = labels.to(device)

      predictions = model(token_ids)

      labels = labels.view(-1, 1)

      loss = loss_fn(predictions, labels.float())

      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()


    return epoch_loss / len(data_loader)

def evaluate(model, device, data_loader, loss_fn):
    model.eval()
    y_true = []
    y_pred = []

    epoch_loss=0

    with torch.no_grad():

      for step, batch in enumerate(tqdm(data_loader, desc="Evaluation...")):
          token_ids,labels = batch

          token_ids = token_ids.to(device)
          labels = labels.to(device)

          predictions = model(token_ids)

          labels = labels.view(-1, 1)

          loss = loss_fn(predictions, labels.float())

          epoch_loss += loss.item()

          predicted_labels = (predictions > 0.5).float()

          y_true.extend(labels.cpu().numpy())
          y_pred.extend(predicted_labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    avg_loss = epoch_loss / len(data_loader)

    return avg_loss, accuracy, precision, recall, f1