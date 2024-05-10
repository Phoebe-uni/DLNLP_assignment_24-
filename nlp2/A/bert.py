'''
Written by YYF.
'''


import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn')

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim

import transformers
from transformers import BertTokenizer, BertModel, BertConfig

import tokenizers

import nltk

from sklearn.model_selection import train_test_split

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
MAX_LENGTH = 110
max_len = MAX_LENGTH
def bert_main():
    global max_len
    train_df = pd.read_csv('./Datasets/train.csv')
    test_df = pd.read_csv('./Datasets/test.csv')
    info = f'./A/output/bert_info.txt'    ## write the terminal info into the text
    gc.collect()
    torch.cuda.empty_cache()
    EPOCHS = 8
    SAVE_MODEL_PATH = f'./A/output/bert.pth'

    ## get the tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        '../input/bert-base-uncased/vocab.txt',
        do_lower_case=True)

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    max_len = get_max_len(train_df, tokenizer)

    ## get the train and validation data
    train_df, val_df = train_test_split(train_df, test_size=0.2)
    train_dataset = TextDataset(train_df, tokenizer, max_len=max_len, is_label=True)
    val_dataset = TextDataset(val_df, tokenizer, max_len=max_len, is_label=True)
    test_dataset = TextDataset(test_df, tokenizer, max_len=max_len, is_label=False)

    # convert data to bert model dataset
    dataloader_dict = train_val_dataloaders(train_dataset, val_dataset)
    test_loader = get_test_loader(test_dataset)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print('device used with : ', device)


    file = open(info, "w")
    strs = f"======== Bert Training and Testing ... ========"
    print(strs)
    file.write(f'{strs}\n')

    ## use the bert model
    model = BertModel()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
    criterion = fn_loss

    ## Training ...
    train_loss, val_loss, jac_train, jac_val = ModelTraining(
        model,
        device,
        dataloader_dict,
        criterion,
        optimizer,
        EPOCHS,
        SAVE_MODEL_PATH,
        file, tokenizer)

    file.write("\n---- Now save the train loss, val loss, train jaccard, val jaccard ----\n")
    file.write("\nepoch, train loss, val loss, train jaccard, val jaccard\n")

    ## write the loss and jaccard to info file.
    for i in range(len(train_loss)):
        strs = "{0}, {tr_loss:.3f}, {vl_loss:.3f}, {tr_jac:.3f}, {vl_jac}\n".format(i + 1, tr_loss=train_loss[i],
                                                                                    vl_loss=val_loss[i],
                                                                                    tr_jac=jac_train[i],
                                                                                    vl_jac=jac_val[i])
        print(strs)
        file.write(strs)

    strs = '\n======== End Berta Training and Testing ========'
    print(strs)
    file.write(f'{strs}\n')
    file.close()

    ## plot the loss and jaccard fig.
    plotfig(train_loss, val_loss, jac_train, jac_val)

    #Now Model Testing(Prediction)...
    predictions = ModelTesting(device, test_loader, tokenizer)

    ## write the test(prediction) data to submission.csv
    submission = pd.read_csv(f'./Dataset/sample_submission.csv')
    submission['selected_text'] = predictions
    submission.to_csv(f'./A/output/submission.csv', index=False)

def get_max_len(df, tokenizer):
    max_len = 0
    for text in df['text']:

        # Tokenize the text and add special tokens i.e `[CLS]` and `[SEP]`
        input_ids = tokenizer.encode(text, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    return max_len


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LENGTH, is_label=False):
        self.df = df
        self.max_len = max_len + 3  # 3 means `[CLS]` and sentiment and `[SEP]`
        self.is_label = is_label
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        global data
        data = {}
        row = self.df.iloc[index]

        ids, masks, token_type = self.get_bert_tokenize(row)
        data['input_ids'] = ids
        data['attention_masks'] = masks
        data['token_type_ids'] = token_type

        # Text / Selected Text Decode
        data['text'] = self.tokenizer.decode(ids)

        if self.is_label:
            start_idx, end_idx = self.get_label_idx(data, row)
            data['start_index'] = start_idx
            data['end_index'] = end_idx

        return data

    def get_label_idx(self, data, row):
        # get lavel ids
        global start_index
        global end_index

        text_id = self.tokenizer.encode(
            row['selected_text'],
            add_special_tokens=False,
        )
        label_len = len(text_id)

        # get start index / end index
        for i in range(self.max_len):
            if data['input_ids'][i] == text_id[0]:
                if data['input_ids'][i + label_len - 1] == text_id[-1]:
                    start_index = i
                    end_index = i + label_len - 1
                    break

        return torch.tensor(start_index), torch.tensor(end_index)

    def get_bert_tokenize(self, row):

        text = row['text']
        sentiment = row['sentiment']

        encoded = self.tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze()
        attention_masks = encoded['attention_mask'].squeeze()
        token_type_ids = encoded['token_type_ids'].squeeze()

        return input_ids, attention_masks, token_type_ids


def train_val_dataloaders(train_dataset, val_dataset, batch_size=8):
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)

    dataloaders_dict = {"train": train_loader, "val": val_loader}
    return dataloaders_dict


def get_test_loader(dataset, batch_size=32):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)

    return loader


class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        config = BertConfig.from_pretrained(
            './A/input/config.json', output_hidden_states=True)
        self.bert = transformers.BertModel.from_pretrained(
            './A/input/pytorch_model.bin', config=config)
        self.LSTM = nn.LSTM(self.hidden_size * 2, 128)
        self.hidden_size = self.bert.config.hidden_size
        self.layer = nn.Sequential(nn.Linear(128, 64), nn.Dropout(0.2), )

        # The output will have two dimensions ("start_logits", and "end_logits")
        self.FC = nn.Linear(64, 2)
        torch.nn.init.normal_(self.FC.weight, std=0.02)

    def forward(self, ids, mask, token):
        # Return the hidden states from the BERT backbone
        out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token
        )

        out = torch.cat((out[2][-1], out[2][-2]), dim=-1)

        out, _ = self.LSTM(out)
        out = self.layer(out)
        logits = self.FC(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

def fn_loss(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss


def get_selected_text(text_encode, start_idx, end_idx, tokenizer):
    text_encode = text_encode[start_idx: end_idx + 1]
    selected_text = tokenizer.decode(text_encode)
    return selected_text


def get_original_text(text_encode, tokenizer):
    text_encode = text_encode[3:]
    for i, encode in enumerate(text_encode):
        if encode == 102:
            last_index = i
            break
    return tokenizer.decode(text_encode[:last_index])


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    d = (len(a) + len(b) - len(c))
    if d != 0:
         return float(len(c) / d)
    else:
         return 0.0


def compute_jaccard_score(text_encode, start_idx, end_idx, start_logits, end_logits, tokenizer):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)

    if start_pred > end_pred:
        pred = get_original_text(text_encode, tokenizer)

    else:
        pred = get_selected_text(text_encode, start_pred, end_pred, tokenizer)

    true = get_selected_text(text_encode, start_idx, end_idx, tokenizer)
    return jaccard(true, pred)


def ModelTraining(model, device, dataloaders_dict, criterion, optimizer, num_epochs, filename, file, tokenizer):
    train_loss = []
    val_loss = []
    jac_train = []
    jac_val = []

    model.to(device)

    for epoch in range(num_epochs):
        for proc in ['train', 'val']:
            if proc == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_jaccard = 0.0

            for j, data in enumerate((dataloaders_dict[proc])):
                ids = data['input_ids'].to(device, dtype=torch.int64)
                masks = data['attention_masks'].to(device, dtype=torch.int64)
                token = data['token_type_ids'].to(device, dtype=torch.int64)
                start_idx = data['start_index'].to(device, dtype=torch.int64)
                end_idx = data['end_index'].to(device, dtype=torch.int64)

                optimizer.zero_grad()

                with torch.set_grad_enabled(proc == 'train'):

                    start_logits, end_logits = model(ids, masks, token)

                    loss = criterion(start_logits, end_logits, start_idx, end_idx)

                    if proc == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(ids)

                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

                    for i in range(len(ids)):
                        jaccard_score = compute_jaccard_score(
                            ids[i],
                            start_idx[i],
                            end_idx[i],
                            start_logits[i],
                            end_logits[i], tokenizer)
                        epoch_jaccard += jaccard_score

            epoch_loss = epoch_loss / len(dataloaders_dict[proc].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[proc].dataset)
            if proc == 'train':
                train_loss.append(epoch_loss)
                jac_train.append(epoch_jaccard)
            else:
                val_loss.append(epoch_loss)
                jac_val.append(epoch_jaccard)
            strs = 'Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(
                epoch + 1, num_epochs, proc, epoch_loss, epoch_jaccard)

            print(strs)
            file.write(f'{strs}\n')

    torch.save(model.state_dict(), filename)
    return train_loss, val_loss, jac_train, jac_val

def plotfig(train_loss_list, vl_loss_list, train_acc_list, vl_acc_list ):
    '''
    plot four figures:  train loss, validation loss, train acc, validation acc vs epochs
    Args:
        train_loss_list: training loss for all epochs
        vl_loss_list: validation loss for all epochs
        train_acc_list: training accuracy for all epochs
        vl_acc_list: validation accuracy for all epochs

    Returns: 2 figures

    '''
    x1 = range(1, len(train_loss_list) + 1)  # set start value to 1
    x2 = range(1, len(vl_loss_list)+1 )
    x3 = range(1, len(train_acc_list) + 1)
    x4 = range(1, len(vl_acc_list)+1)
    y1 = train_loss_list
    y2 = vl_loss_list
    y3 = train_acc_list
    y4 = vl_acc_list

    fig, axs = plt.subplots(2, 1, figsize=(8, 12))

    axs[0].plot(x1, y1, 'bo-',label='Training Loss')
    axs[0].plot(x2, y2, 'ro-',label='Validation Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc='best')

    axs[1].plot(x3, y3, 'bo-',label='Training Jaccard')
    axs[1].plot(x4, y4, 'ro-',label='Validation Jaccard')
    axs[1].set_title('Training and Validation Jaccard')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Jaccard Score')
    axs[1].legend(loc='best')

    plt.legend(loc='best')
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f'fig_bert.jpg')
    #plt.show()
    plt.clf()  # Clear figure
    plt.cla()  # Clear axes
    plt.close()

def ModelTesting(device, test_df, tokenizer):

    predictions = []
    model = BertModel()
    model.to(device)
    model.load_state_dict(torch.load(f'bert.pth'))
    model.eval()
    for data in test_df:
        ids = data['input_ids'].to(device, dtype=torch.int64)
        masks = data['attention_masks'].to(device, dtype=torch.int64)
        token = data['token_type_ids'].to(device, dtype=torch.int64)

        start_logits = []
        end_logits = []
        with torch.no_grad():
            start_logit, end_logit = model(ids, masks, token)
            start_logits = torch.softmax(start_logit, dim=1).cpu().detach().numpy()
            end_logits = torch.softmax(end_logit, dim=1).cpu().detach().numpy()

        for i in range(len(ids)):
            start_pred = np.argmax(start_logits[i])
            end_pred = np.argmax(end_logits[i])
            if start_pred > end_pred:
                pred = get_original_text(ids[i], tokenizer)
            else:
                pred = get_selected_text(ids[i], start_pred, end_pred, tokenizer)
            predictions.append(pred)
    return predictions


if __name__ == '__main__':
    bert_main()