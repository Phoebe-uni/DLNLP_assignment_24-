'''
Written by YYF.
'''
import torch
import pandas as pd
import numpy as np
import re
import random

import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.optim import lr_scheduler

from tqdm.autonotebook import tqdm

#nltk
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
import string

from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
import transformers
import tokenizers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaModel, RobertaConfig
from torch.utils.data import Dataset, DataLoader

from transformers import logging
logging.set_verbosity_warning()
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

MAX_LENGTH = 110
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
EPOCHS = 8

train_loss = []
val_loss = []
jac_train = []
jac_val = []

def roberta_main():
    global train_loss
    global val_loss
    global jac_train
    global jac_val

    info = './B/output/roberta_info.txt'  ## write the terminal info into the text

    file = open(info, "w")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    strs = f"======== Roberta Training and Testing ... ========"
    print(strs)
    file.write(f'{strs}\n')

    train_df = pd.read_csv('./Datasets/train.csv')
    train_df.drop(314, inplace=True)  # This row was found to have 'nan' values, so dropping it
    train_df.reset_index(drop=True, inplace=True)
    train_df['text'] = train_df['text'].astype(str)
    train_df['selected_text'] = train_df['selected_text']

    test_df = pd.read_csv('./Datasets/test.csv').reset_index(drop=True)
    test_df['text'] = test_df['text'].astype(str)

    PATH = "./B/input"
    ## get the tokenizer
    tokenizer = ByteLevelBPETokenizer(
        vocab=f'{PATH}/vocab.json',
        merges=f'{PATH}/merges.txt',
        add_prefix_space=True,
        lowercase=True)
    tokenizer.enable_truncation(max_length=512)  # since length cannot be set, use enable_truncation() instead

    val1 = tokenizer.encode('negative').ids
    val2 = tokenizer.encode('positive').ids
    val3 = tokenizer.encode('neutral').ids
    strs = "negative, positive, neutral ids is: {},{},{}".format(val1, val2, val3)
    print(strs)
    file.write(f'{strs}\n')


    seed = 777
    set_seed(seed)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    ##Model Training...
    for fold, (idxTrain, idxVal) in enumerate(skf.split(train_df, train_df.sentiment), start=1):
        strs = '#' * 10
        print(strs)
        file.write(f'{strs}\n')

        strs = '### FOLD %i' % (fold)
        print(strs)
        file.write(f'{strs}\n')

        strs = '#' * 10
        print(strs)
        file.write(f'{strs}\n')

        #use Roberta Model
        model = RobertaModel()
        optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, correct_bias=False)
        dataloaders_dict = train_val_dataloaders(train_df, idxTrain, idxVal, tokenizer, batch_size=TRAIN_BATCH_SIZE)
        num_training_steps = int(len(train_df) / EPOCHS * TRAIN_BATCH_SIZE)
        # warmup_proportion = float(num_warmup_steps) / float(num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # default #use a linear scheduler with no warmup steps
            num_training_steps=num_training_steps
        )

        ModelTraining(model, dataloaders_dict, optimizer, EPOCHS,
            scheduler, device, f'./B/output/bert_fold{fold}.pth', file, tokenizer )

    ## plot the loss and jaccard fig.
    plotfig(train_loss,val_loss, jac_train, jac_val)

    ## Model test(prediction)
    predictions = ModelTesting(device, test_df, tokenizer, skf)
    sub_df = pd.read_csv('./Datasets/sample_submission.csv')
    sub_df['selected_text'] = predictions
    # post-processing trick
    sub_df['selected_text'] = sub_df['selected_text'].apply(
        lambda x: x.replace('!!!!', '!') if len(x.split()) == 1 else x)
    sub_df['selected_text'] = sub_df['selected_text'].apply(
        lambda x: x.replace('..', '.') if len(x.split()) == 1 else x)
    sub_df['selected_text'] = sub_df['selected_text'].apply(
        lambda x: x.replace('...', '.') if len(x.split()) == 1 else x)
    sub_df[['textID', 'selected_text']].to_csv('./B/output/submission.csv', index=False)
    pd.set_option('max_colwidth', 60)
    strs = "{}".format(sub_df.sample(25))
    print(strs)
    file.write(f'{strs}\n')

    file.write("\n---- Now save the train loss, val loss, train jaccard, val jaccard ----\n")
    file.write("\nepoch, train loss, val loss, train jaccard, val jaccard\n")

    for i in range(len(train_loss)):
        strs = "{0}, {tr_loss:.3f}, {vl_loss:.3f}, {tr_jac:.3f}, {vl_jac}\n".format(i+1, tr_loss = train_loss[i],
                vl_loss = val_loss[i], tr_jac = jac_train[i], vl_jac = jac_val[i])
        file.write(strs)

    print('\n======== Now saving the the loss and jaccard fig ======== ')

    plotfig(train_loss,val_loss, jac_train, jac_val)

    strs = '\n======== End Roberta Training and Testing ========'
    print(strs)
    file.write(f'{strs}\n')
    file.close()

class TextDataset(Dataset):

    def __init__(self, df, tokenizer, max_length = MAX_LENGTH):
        # data loading
        self.df = df
        self.selected_text = "selected_text" in df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # len(dataset) i.e., the total number of samples
        return len(self.df)

    def get_data(self, row):
        # processing the data
        text = " " + " ".join(row.text.lower().split())  # clean the text
        encoded_input = self.tokenizer.encode(text)  # the sentence to be encoded

        sentiment_id = {
            'positive': 1313,
            'negative': 2430,
            'neutral': 7974
        }  # stating the ids of the sentiment values

        # print ([list((i, encoded_input[i])) for i in range(len(encoded_input))])
        '''
        # The input_ids are the sentence or sentences represented as tokens. 
        # There are a few BERT special tokens that one needs to take note of:

        # [CLS] - Classifier token, value: [101] 
        # [SEP] - Separator token, value: [102]
        # [PAD] - Padding token, value: 0

        # Bert expects every row in the input_ids to have the special tokens included as follows:

        # For one sentence as input:
        # [CLS] ...word tokens... [SEP]

        # For two sentences as input:
        # [CLS] ...sentence1 tokens... [SEP]..sentence2 tokens... [SEP]
        '''

        input_ids = [101] + [sentiment_id[row.sentiment]] + [102] + encoded_input.ids + [102]

        '''
        id: unique identifier for each token
        offset: starting and ending point in a sentence
        '''

        # ID offsets
        offsets = [(0, 0)] * 3 + encoded_input.offsets + [
            (0, 0)]  # since first 3 are [CLS] ...sentiment tokens... [SEP]

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += ([0] * pad_len)
            offsets += ([(0, 0)] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        masks = torch.where(input_ids != 0, torch.tensor(1), torch.tensor(0))
        '''
        # The attention mask has the same length as the input_ids(or token_type_ids). 
        # It tells the model which tokens in the input_ids are words and which are padding. 
        # 1 indicates a word (or special token) and 0 indicates padding.

        # For example:
        # Tokens: [101, 7592, 2045, 1012,  102,    0,    0,    0,    0,    0]
        # Attention mask: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        '''

        masks = torch.tensor(masks, dtype=torch.long)
        offsets = torch.tensor(offsets, dtype=torch.long)
        return input_ids, masks, text, offsets

    def get_target_ids(self, row, text, offsets):
        # preparing data only for the training
        selected_text = " " + " ".join(row.selected_text.lower().split())

        string_len = len(selected_text) - 1

        idx0 = None
        idx1 = None

        for ind in (position for position, line in enumerate(text) if line == selected_text[1]):
            if " " + text[ind: ind + string_len] == selected_text:
                idx0 = ind
                idx1 = ind + string_len - 1
                break

        char_targets = [0] * len(text)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1
        '''
        char_targets only give 1 to the part of the selected_text within a text
        e.g: [0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        Here, for the length of the text, [1] is only at the index position of the selected text.
        This helps us get the start and end indices of the selected text in the next stage.
        '''
        # Start and end tokens
        target_idx = []
        for k, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                try:
                    target_idx.append(k)
                except:
                    continue

        targets_start = target_idx[0]
        targets_end = target_idx[-1]

        return selected_text, targets_start, targets_end

    def __getitem__(self, index):  # addressing each row by its index
        # dataset[index] i.e., generates one sample of data
        data = {}
        row = self.df.iloc[index]

        ids, masks, text, offsets = self.get_data(row)
        data['ids'] = ids
        data['masks'] = masks
        data['text'] = text
        data['offsets'] = offsets
        data['sentiment'] = row.sentiment

        if self.selected_text:  # checking if selected text exists
            # This part only exists in the training
            selected_text, start_index, end_index = self.get_target_ids(row, text, offsets)
            data['start_index'] = start_index
            data['end_index'] = end_index
            data['selected_text'] = selected_text

        return data


class RobertaModel(nn.Module):

    def __init__(self):
        super(RobertaModel, self).__init__()
        config = RobertaConfig.from_pretrained(
            './B/input/config.json', output_hidden_states=True)
        self.roberta = RobertaModel.from_pretrained(
            './B/input/pytorch_model.bin', config=config)

        for param in self.roberta.parameters():
            param.requires_grad = True

        self.drop0 = nn.Dropout(0.2)
        self.l0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        # Multiplied by 2 since the forward pass concatenates the last two hidden representation layers
        self.drop1 = nn.Dropout(config.hidden_dropout_prob)
        self.l1 = nn.Linear(config.hidden_size, 2)  # The output will have two dimensions- start and end logits
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        torch.nn.init.normal_(self.l0.bias, 0)

    def forward(self, ids, masks):  # , token_type_ids
        # Return the hidden states from the RoBERTa backbone
        # Type: torch tensor
        last_hidden_state, pooled_output, hidden_states = self.roberta(input_ids=ids, attention_mask=masks,
                                                                       return_dict=False)
        # input_ids.shape and attention_mask.shape both will be of the size (batch size x seq length)
        # print(last_hidden_state.shape) : torch.Size([24, 100, 768])
        # 768: This is the number of hidden units in the feedforward-networks. We can verify that by checking the config.
        '''
        About the parameters:

        input_ids (torch.LongTensor of shape (batch_size, sequence_length)) –
        Indices of input sequence tokens in the vocabulary.
        Indices can be obtained using Tokenizer. See transformers.PreTrainedTokenizer.encode() and transformers.PreTrainedTokenizer.__call__() for details.

        attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional) –
        Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
        1 for tokens that are not masked,
        0 for tokens that are masked.

        '''

        # Concatenate the last two hidden states
        out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
        # out = torch.mean(out, 0) # take the mean along axis 0

        # adding dropouts and linear layers
        out = self.drop0(out)
        out = F.relu(self.l0(out))
        out = self.drop1(out)
        out = self.l1(out)

        # splitting the tensor into two logits
        start_logits, end_logits = out.split(1, dim=-1)  # dimension along which to split the tensor.
        # Return a tensor with all the dimensions of input of size 1 removed, for both the logits.
        start_logits = start_logits.squeeze()  # Squeezing a tensor removes the dimensions or axes that have a length of one
        end_logits = end_logits.squeeze()

        return start_logits, end_logits

def train_val_dataloaders(df, train_idx, val_idx, tokenizer, batch_size):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TextDataset(train_df, tokenizer, MAX_LENGTH),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        TextDataset(val_df, tokenizer, MAX_LENGTH),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict

def test_loader(df, tokenizer, batch_size=TEST_BATCH_SIZE):
    loader = torch.utils.data.DataLoader(
        TextDataset(df, tokenizer, MAX_LENGTH),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
    return loader

def fn_loss(start_logits, end_logits, start_positions, end_positions):
    # calculating cross entropy losses for both the start and end logits
    loss = nn.CrossEntropyLoss(reduction='mean') # for a multi-class classification problem
    start_loss = loss(start_logits, start_positions)
    end_loss = loss(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss

def get_selected_text(text, start_idx, end_idx, offsets):
    if end_idx < start_idx:
        end_idx = start_idx
    select_text = ""
    for idx in range(start_idx, end_idx + 1):
        select_text += text[offsets[idx][0]: offsets[idx][1]]
        if (idx + 1) < len(offsets) and offsets[idx][1] < offsets[idx + 1][0]:
            select_text += " "
    return select_text

# evaluation metric
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    d = (len(a) + len(b) - len(c))
    if d != 0:
         return float(len(c) / d)
    else:
         return 0.0

def find_jaccard_score(text, selected_text, sentiment, offsets, start_logits, end_logits, tokenizer): #start_idx, end_idx
    start_pred = np.argmax(start_logits) # Predicted start index using argmax
    end_pred = np.argmax(end_logits) # Predicted end index using argmax
    if (end_pred <= start_pred) or sentiment == 'neutral' or len(text.split()) < 2:
        enc = tokenizer.encode(text)
        prediction = tokenizer.decode(enc.ids[start_pred-1:end_pred])
    else:
        prediction = get_selected_text(text, start_pred, end_pred, offsets)
    true = selected_text
    #true = get_selected_text(text, start_idx, end_idx, offsets)
    return jaccard(true, prediction), prediction


def ModelTraining(model, dataloaders_dict, optimizer, num_epochs, scheduler, device, filename, file, tokenizer):
    '''
    Train pytorch model on a single pass through the data loader.

        This function is built with reusability in mind: it can be used as is as long
        as the `dataloader` outputs a batch in dictionary format that can be passed
        straight into the model - `model(**batch)`.

      Some of the arguments:

      dataloaders_dict (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.

      optimizer_ (:obj:`transformers.optimization.AdamW`):
          Optimizer used for training.

      num_epochs:  training epochs

      scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
          PyTorch scheduler.

      device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.

      filename: save model state dict to this filename;

      file:  write the info of train and validate status to file;

      tokenizer: tokenizer of the data
    '''
    # Set device as `cuda` (GPU)
    global train_loss
    global val_loss
    global jac_train
    global jac_val
    model.to(device)
    for epoch in range(num_epochs):
        for key in ['train', 'val']:
            if key == 'train':
                model.train()
                dataloaders = dataloaders_dict['train']
            else:
                model.eval()
                dataloaders = dataloaders_dict['val']

            epoch_loss = 0.0
            epoch_jaccard = 0.0

            # Set tqdm to add loading screen and set the length
            loader = tqdm(dataloaders, total=len(dataloaders))
            # print(len(dataloaders))

            # loop over the data iterator, and feed the inputs to the network
            # Train the model on each batch
            for (idx, data) in enumerate(loader):
                ids = data['ids']
                masks = data['masks']
                text = data['text']
                offsets = data['offsets'].numpy()
                start_idx = data['start_index']
                end_idx = data['end_index']
                sentiment = data['sentiment']

                model.zero_grad()
                optimizer.zero_grad()

                ids = ids.to(device, dtype=torch.long)
                masks = masks.to(device, dtype=torch.long)
                start_idx = start_idx.to(device, dtype=torch.long)
                end_idx = end_idx.to(device, dtype=torch.long)

                with torch.set_grad_enabled(key == 'train'):

                    start_logits, end_logits = model(ids, masks)

                    loss = fn_loss(start_logits, end_logits, start_idx, end_idx)

                    if key == 'train':
                        if idx != 0:
                            loss.backward()  # Perform a backward pass to calculate the gradients
                        optimizer.step()  # Update parameters and take a step using the computed gradient
                        scheduler.step()  # Update learning rate schedule

                        # Clip the norm of the gradients to 1.0.
                        # This is to help prevent the "exploding gradients" problem.
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    epoch_loss += loss.item() * len(ids)

                    # Move logits to CPU
                    # detaching these outputs so that the backward passes stop at this point
                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

                    selected_text = data['selected_text']

                    filtered_sentences = []
                    for i, t_data in enumerate(text):
                        # for i in range(len(ids)):
                        jaccard_score, filtered_output = find_jaccard_score(
                            t_data,
                            selected_text[i],
                            sentiment[i],
                            offsets[i],
                            start_logits[i],
                            end_logits[i], tokenizer)
                        epoch_jaccard += jaccard_score
                        filtered_sentences.append(filtered_output)

            # Calculate the average loss over the training data
            epoch_loss = epoch_loss / len(dataloaders.dataset)
            # Calculate the average jaccard score over the training data
            epoch_jaccard = epoch_jaccard / len(dataloaders.dataset)

            strs = 'Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(
                epoch + 1, num_epochs, key, epoch_loss, epoch_jaccard)
            print(strs)
            file.write(f'{strs}\n')

            # Store the loss value for plotting the learning curve.
            if key == 'train':
                train_loss.append(epoch_loss)
                jac_train.append(epoch_jaccard)

            else:
                val_loss.append(epoch_loss)
                jac_val.append(epoch_jaccard)

    torch.save(model.state_dict(), filename)
    #return train_loss,val_loss, jac_train, jac_val


def set_seed(seed):
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

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
    plt.savefig('./B/output/fig_roberta.jpg')
    plt.clf()  # Clear figure
    plt.cla()  # Clear axes
    plt.close()


def ModelTesting(device, test_df, tokenizer, skf):

    t_loader = test_loader(test_df, tokenizer)
    predictions = []
    models = []
    for fold in range(skf.n_splits):
        model = RobertaModel()
        model.to(device)
        model.load_state_dict(torch.load(f'./B/output/bert_fold{fold + 1}.pth'))
        model.eval()
        models.append(model)

    loader = tqdm(t_loader, total=len(t_loader))
    for (idx, data) in enumerate(loader):
        ids = data['ids'].to(device)
        masks = data['masks'].to(device)
        text = data['text']
        offsets = data['offsets'].numpy()

        start_logits = []
        end_logits = []
        for model in models:
            with torch.no_grad():
                output = model(ids, masks)
                start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

        start_logits = np.mean(start_logits, axis=0)
        end_logits = np.mean(end_logits, axis=0)

        for i, t_data in enumerate(text):
            start_pred = np.argmax(start_logits[i])
            end_pred = np.argmax(end_logits[i])
            if start_pred >= end_pred:
                enc = tokenizer.encode(t_data)
                prediction = tokenizer.decode(enc.ids[start_pred - 1:end_pred])
            else:
                prediction = get_selected_text(t_data, start_pred, end_pred, offsets[i])
            predictions.append(prediction)
    return predictions

if __name__ == '__main__':
    roberta_main()