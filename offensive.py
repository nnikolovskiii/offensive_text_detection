import argparse
from torch.utils.data import DataLoader

from modeling.modeling_rnn import RNN
from preprocessing.preprocess import load_datasets, create_vocab, tokenize_df
from utils.data_utils import MyDataset, collate_batch, train, evaluate
import math
import torch
import torch.nn as nn

def main(args):
    df_train, df_val, df_test = load_datasets()
    vocab = create_vocab(df_train)
    df_train = tokenize_df(df_train, vocab)
    df_val = tokenize_df(df_val, vocab)
    df_test = tokenize_df(df_test, vocab)

    d_train = MyDataset(df=df_train)
    d_test = MyDataset(df=df_test)
    d_val = MyDataset(df=df_val)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_size = len(vocab)+1

    train_dataloader = DataLoader(d_train, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda batch: collate_batch(batch))
    test_dataloader = DataLoader(d_test, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=lambda batch: collate_batch(batch))
    val_dataloader = DataLoader(d_val, batch_size=args.batch_size, shuffle=True,
                                collate_fn=lambda batch: collate_batch(batch))

    model = RNN(vocab_size, args.emb_dim, args.hid_dim, args.output_dim, args.n_layers, args.dropout)
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_loss = float('inf')
    for epoch in range(args.N_EPOCHS):

        train_loss = train(model, device, train_dataloader, optimizer, loss_fn)
        loss, accuracy, precision, recall, f1 = evaluate(model, device, val_dataloader, loss_fn)

        if loss < best_valid_loss:
            best_valid_loss = loss
            torch.save(model.state_dict(), args.model_save_path)

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {loss:.3f} |  Val. PPL: {math.exp(loss):7.3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an RNN model for offensive text detection.')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--emb_dim', type=int, default=100, help='Dimension of word embeddings')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimension of output (1 or 0)')
    parser.add_argument('--hid_dim', type=int, default=256, help='Dimension of hidden state')
    parser.add_argument('--N_EPOCHS', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default=r'data/tut1-model.pt', help='Path to save the trained model')

    args = parser.parse_args()

    main(args)
