import torch
import torch.nn as nn
import math
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
import re
from transformers import BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ot
from torchmetrics.classification import F1Score

from dataloader import *
from transformer import *


# IMPORT DATA
def import_data():
    # load data
    data = pd.read_csv("../Data/IMDB Dataset.csv", encoding='ISO-8859-1')

    # convert string label to binary (int) label (spam: 1, non-spam: 0)
    data["sentiment"] = data['sentiment'].apply(lambda x: int(x == "positive"))

    return data


def preproc(dat):
    data = dat.copy()

    # proc
    data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x: x.replace("<br />", " "))  # remove break symbols
    data.iloc[:, 0] = data.iloc[:, 0].apply(lambda x: x.lower())  # all lower case

    # only use short sentences
    data["len"] = data.iloc[:, 0].apply(lambda x: len(x.split()))
    data = data[data["len"] < 256]

    # tokenize data (split sentence into tokens)
    tokenizer = Tokenizer()
    print("Tokenizing the data...")
    data["len"] = data.iloc[:, 0].apply(lambda x: len(tokenizer.tokenize(x)))

    # only use short sentences
    data = data[data["len"] < 256]

    # dataset size
    print("Length of the data : ", len(data))

    # further proc
    data.iloc[:, 0] = data.iloc[:, 0].apply(
        lambda row: re.sub("[^A-Za-z]+", " ", row)).apply(tokenizer.tokenize)

    # reset index (we deleted some rows)
    data = data.reset_index(drop=True)

    return data


def build_generators(train, test, device, batch_size=512):
    """Builds training and test generators from specific datasets"""
    # build vocab
    vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2}
    for item in train.iloc[:, 0]:
        for word in item:
            if word not in vocab:
                vocab[word] = len(vocab)
    pad_idx = vocab['__PAD__']
    voc_size = len(vocab)
    print("Vocabulary Size : ", voc_size)

    # create embedding
    embedding = torch.nn.Embedding(voc_size, 16)

    # dataframe to tensor
    train_y = torch.tensor(train.iloc[:, 1].values.astype(np.float32), device=device)
    test_y = torch.tensor(test.iloc[:, 1].values.astype(np.float32), device=device)

    unk_ID = vocab["__UNK__"]

    train_x_tensor = []
    for idx, text_corpus in enumerate(tqdm(train.iloc[:, 0])):
        foo = []
        for token in text_corpus:
            word_ID = vocab.get(token, unk_ID)
            foo.append(word_ID)
        while len(foo) < 256:
            foo.append(vocab["__PAD__"])
        train_x_tensor.append(foo)

    test_x_tensor = []
    for idx, text_corpus in enumerate(tqdm(test.iloc[:, 0])):
        foo = []
        for token in text_corpus:
            word_ID = vocab.get(token, unk_ID)
            foo.append(word_ID)
        while len(foo) < 256:
            foo.append(vocab["__PAD__"])
        test_x_tensor.append(foo)

    train_x = torch.tensor(train_x_tensor, device=device)
    test_x = torch.tensor(test_x_tensor, device=device)

    train = torch.utils.data.TensorDataset(train_x, train_y)
    test = torch.utils.data.TensorDataset(test_x, test_y)

    train_iterator = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_iterator = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    # check imbalance
    check_imbalance(train_iterator, name='train set')
    check_imbalance(test_iterator, name='test set')

    print('Dataset initializing done')
    return train_iterator, test_iterator, voc_size, pad_idx, embedding


def train_test_val_split(data, device, batch_size=512):
    # tokenizer (split sentence into tokens)
    tokenizer = Tokenizer()
    loader = DataLoader(tokenize=tokenizer.tokenize)

    train, valid, test = loader.make_dataset(data)
    train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                         batch_size=batch_size,
                                                         device=device)

    # NLP stuff
    vocab = loader.get_vocab(train.iloc[:, 0])
    voc_size = len(vocab)
    pad_idx = vocab['__PAD__']
    print("Vocabulary Size : ", voc_size)

    return train_iter, valid_iter, test_iter, voc_size, pad_idx


def check_imbalance(iterator, name=None):
    all_y = np.array([])
    for i, batch in enumerate(iterator):
        _, trg = batch  # X, y
        all_y = np.append(all_y, trg.cpu().numpy())
    if name is not None:
        print(f'Positive labels ratio ({name}):', np.mean(all_y))
    else:
        print('Positive labels ratio:', np.mean(all_y))


# EXPORT DATA
def save_model(model, name: str):
    torch.save(model, f'./Models/{name}')


def save_history(history, name: str):
    with open(f'./Models/{name}.txt', 'w') as dat:
        dat.write(str(history))


# MODEL FUNCTIONS
def new_model(embedding, pad_idx, voc_size, device, max_len=256, d_model=16, ffn_hidden=32, n_head=1, n_layers=1,
              drop_prob=0.5):
    """Initializes a new model"""
    model = TransformerClassifier(embedding=embedding,
                                  src_pad_idx=pad_idx,
                                  enc_voc_size=voc_size,
                                  max_len=max_len,
                                  d_model=d_model,
                                  ffn_hidden=ffn_hidden,
                                  n_head=n_head,
                                  n_layers=n_layers,
                                  drop_prob=drop_prob,
                                  device=device)
    return model.to(device)  # put on CPU/GPU


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


def plot_training(history, marker=None):
    # put everything on cpu
    for key, value in history.items():
        history[key] = [element.cpu() if isinstance(element, torch.Tensor) else element for element in value]

    plt.subplots_adjust(left=0.1,
                        bottom=0.01,
                        right=1.5,
                        top=0.6,
                        wspace=0.4,
                        hspace=0.4)

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('Training loss')

    # vertical line for marking best epoch
    if marker is not None:
        y_min = min(history['train_loss'] + history['val_loss'])
        y_max = max(history['train_loss'] + history['val_loss'])
        plt.vlines(x=marker, ymin=y_min, ymax=y_max, color='red')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('Training metric')

    # vertical line for marking best epoch
    if marker is not None:
        y_min = min(history['train_acc'] + history['val_acc'])
        y_max = max(history['train_acc'] + history['val_acc'])
        plt.vlines(x=marker, ymin=y_min, ymax=y_max, color='red')

    plt.show()


def validation(model, iterator, criterion, device):
    # set model into evaluation mode
    model.eval()

    # validation
    # loss, metrics for current epoch
    val_epoch_loss = 0
    val_epoch_accuracy = 0
    # for F1 score we don't use batches
    labels_val = []
    preds_val = []
    f1_scorer = F1Score(task='binary').to(device)

    with torch.no_grad():  # stop graph
        # batches
        for i, batch in enumerate(iterator):
            src = batch[0]  # X
            trg = batch[1]  # y
            src, trg = torch.tensor(src).to(device), torch.tensor(trg).to(device)  # put to cpu/gpu
            output = model(src)
            y_pred = torch.argmax(output, dim=-1)  # logits -> labels
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg.to(torch.int64)

            loss = criterion(output_reshape, trg)  # calculate loss
            agreements = torch.eq(y_pred, trg)
            accuracy = torch.mean(agreements.double())  # calculate accuracy
            labels_val.append(trg)
            preds_val.append(y_pred)

            val_epoch_loss += loss.item()
            val_epoch_accuracy += accuracy

    # put to numpy
    labels_val = torch.cat(labels_val)
    preds_val = torch.cat(preds_val)

    # return mean loss w.r.t. batches
    return val_epoch_loss / len(iterator), val_epoch_accuracy / len(iterator), f1_scorer(preds_val, labels_val)


def train(model, iterator, valid_iter, optimizer, criterion, epoch, clip, device, termination_criterion=1e-6):
    # If epoch == 'unrestricted': training until scheduler sets learning rate to 0
    assert isinstance(epoch, int) or epoch == 'unrestricted', f'Invalid epoch: {epoch}'
    unrestricted = False
    if epoch == 'unrestricted':
        epoch = int(1e6)  # large enough
        unrestricted = True

    # set model into training mode
    model.train()

    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # save data - init
    history = {'train_loss': [],
               'val_loss': [],
               'train_acc': [],
               'val_acc': [],
               'learning_rate': []}

    # training
    with tqdm(total=epoch) as pbar:
        for e in range(epoch):
            # termination criterion (unrestricted)
            if unrestricted and optimizer.param_groups[0]['lr'] < termination_criterion:
                print(f'Training has converged after {e} epochs (lr < {termination_criterion})')

                # print training curve
                plot_training(history)

                return history

            # loss, metrics for current epoch
            epoch_loss = 0
            epoch_acc = 0

            # batches
            for i, batch in enumerate(iterator):
                src = batch[0]  # X
                trg = batch[1]  # y
                src, trg = torch.tensor(src).to(device), torch.tensor(trg).to(device)  # put to cpu/gpu
                optimizer.zero_grad()  # reset optimizer
                output = model(src)  # predict
                y_pred = torch.argmax(output, dim=-1)  # logits -> labels
                output_reshape = output.contiguous().view(-1, output.shape[-1])
                trg = trg.to(torch.int64)
                loss = criterion(output_reshape, trg)  # calculate loss
                agreements = torch.eq(y_pred, trg)
                accuracy = torch.mean(agreements.double())  # calculate accuracy
                loss.backward()  # backward pass

                epoch_loss += loss.item()
                epoch_acc += accuracy / len(iterator)

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()  # optimize model

            # validation
            val_loss, val_acc, f1 = validation(model, valid_iter, criterion, device)

            # update scheduler
            scheduler.step(val_loss)

            # save data
            with torch.no_grad():
                current_lr = optimizer.param_groups[0]['lr']
                for key, value in zip(history.keys(),
                                      [epoch_loss / len(iterator), val_loss, epoch_acc, val_acc, current_lr]):
                    history[key].append(value)

            # visualization
            pbar.update(1)
            pbar.set_description(f"Epoch: {e + 1} - Train Loss: {epoch_loss / len(iterator):.4f} /"
                                 f" Validation Loss: {val_loss:.4f} /"
                                 f" Train acc: {epoch_acc:.4f} /"
                                 f" Val acc: {val_acc:.4f} /"
                                 f" Learning Rate : {optimizer.param_groups[0]['lr'] :.4f}")

    # print training curve
    plot_training(history)

    return history


def train_save_best(model, iterator, valid_iter, optimizer, criterion, epoch, clip, device):
    # set model into training mode
    model.train()

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    # save data - init
    history = {'train_loss': [],
               'val_loss': [],
               'train_acc': [],
               'val_acc': [],
               'learning_rate': []}
    best_model = None
    best_model_score = 1e9
    best_model_epoch = 0

    # training
    with tqdm(total=epoch) as pbar:
        for e in range(epoch):
            # loss, metrics for current epoch
            epoch_loss = 0
            epoch_acc = 0

            # batches
            for i, batch in enumerate(iterator):
                src = batch[0]  # X
                trg = batch[1]  # y
                src, trg = torch.tensor(src).to(device), torch.tensor(trg).to(device)  # put to cpu/gpu
                optimizer.zero_grad()  # reset optimizer
                output = model(src)  # predict
                y_pred = torch.argmax(output, dim=-1)  # logits -> labels
                output_reshape = output.contiguous().view(-1, output.shape[-1])
                trg = trg.to(torch.int64)
                loss = criterion(output_reshape, trg)  # calculate loss
                agreements = torch.eq(y_pred, trg)
                accuracy = torch.mean(agreements.double())  # calculate accuracy
                loss.backward()  # backward pass

                epoch_loss += loss.item()
                epoch_acc += accuracy / len(iterator)

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()  # optimize model

            # validation
            val_loss, val_acc, f1 = validation(model, valid_iter, optimizer, criterion, device)

            scheduler.step(val_loss)

            # save data
            with torch.no_grad():
                current_lr = optimizer.param_groups[0]['lr']

                for key, value in zip(history.keys(),
                                      [epoch_loss / len(iterator), val_loss, epoch_acc, val_acc, current_lr]):
                    history[key].append(value)

                # save best model (w.r.t validation loss)
                if val_loss < best_model_score:
                    best_model = model.state_dict()
                    best_model_score = val_loss
                    best_model_epoch = e

            # visualization
            pbar.update(1)
            pbar.set_description(f"Epoch: {e + 1} - Train Loss: {epoch_loss / len(iterator):.4f} /"
                                 f" Validation Loss: {val_loss:.4f} /"
                                 f" Train acc: {epoch_acc:.4f} /"
                                 f" Val acc: {val_acc:.4f} /"
                                 f" Learning Rate : {optimizer.param_groups[0]['lr'] :.4f}")

    # print training curve
    plot_training(history, marker=best_model_epoch)

    return history, best_model, best_model_score


# FUSION
def weight_averaging(*weights):
    device = weights[0].device
    with torch.no_grad():
        sum = torch.zeros(weights[0].shape, device=device)
        for weight in weights:
            sum += weight
    return sum / len(weights)


def linear_averaging(*linears):
    """Averages several linear layers (weights + biases)"""
    with torch.no_grad():
        weights = [linear.weight for linear in linears]
        biases = [linear.bias for linear in linears]

        device = weights[0].device
        linear_averaged = torch.nn.Linear(linears[0].in_features, linears[0].out_features, bias=True).to(device)
        linear_averaged.weight, linear_averaged.bias = torch.nn.Parameter(weight_averaging(*weights)), torch.nn.Parameter(weight_averaging(*biases))

    return linear_averaged


def vanilla_fusion(modelA, modelB, pad_idx, voc_size, embedding, device):
    # init
    n_layers = len(modelA.encoder.layers)
    n_heads = modelA.n_head
    model_fusion = new_model(embedding, pad_idx, voc_size, device, n_head=n_heads, n_layers=n_layers)  # init model

    with torch.no_grad():
        # 1) encoder
        # TODO: smarter method for embedding
        # a) embedding
        '''weights_A = modelA.encoder.emb.tok_emb.weight
        weights_B = modelB.encoder.emb.tok_emb.weight

        weights_fusion = weight_averaging(weights_A, weights_B) # weights seem to be exactly the same?
        model_fusion.encoder.emb.tok_emb.weight = torch.nn.Parameter(weights_fusion)'''

        # b) encoder layers
        for i, _ in enumerate(modelA.encoder.layers):
            # i) self-attention (fuse Q, K, V separately)
            # query
            query_A = modelA.encoder.layers[i].attention.w_q
            query_B = modelB.encoder.layers[i].attention.w_q

            query_fusion = linear_averaging(query_A, query_B)
            model_fusion.encoder.layers[i].attention.w_q = query_fusion

            # key
            key_A = modelA.encoder.layers[i].attention.w_k
            key_B = modelB.encoder.layers[i].attention.w_k

            key_fusion = linear_averaging(key_A, key_B)
            model_fusion.encoder.layers[i].attention.w_k = key_fusion

            # value
            value_A = modelA.encoder.layers[i].attention.w_v
            value_B = modelB.encoder.layers[i].attention.w_v

            value_fusion = linear_averaging(value_A, value_B)
            model_fusion.encoder.layers[i].attention.w_v = value_fusion

            # output
            output_A = modelA.encoder.layers[i].attention.w_concat
            output_B = modelB.encoder.layers[i].attention.w_concat

            output_fusion = linear_averaging(output_A, output_B)
            model_fusion.encoder.layers[i].attention.w_concat = output_fusion

            # ii) layer norm 1
            norm_gamma_A = modelA.encoder.layers[i].norm1.gamma
            norm_gamma_B = modelB.encoder.layers[i].norm1.gamma

            norm_gamma_fusion = weight_averaging(norm_gamma_A, norm_gamma_B)
            model_fusion.encoder.layers[i].norm1.gamma = torch.nn.Parameter(norm_gamma_fusion)

            norm_beta_A = modelA.encoder.layers[i].norm1.beta
            norm_beta_B = modelB.encoder.layers[i].norm1.beta

            norm_beta_fusion = weight_averaging(norm_beta_A, norm_beta_B)
            model_fusion.encoder.layers[i].norm1.beta = torch.nn.Parameter(norm_beta_fusion)

            # iii) feed-forward network
            # layer 1
            linear_A = modelA.encoder.layers[i].ffn.linear1
            linear_B = modelB.encoder.layers[i].ffn.linear1

            linear_fusion = linear_averaging(linear_A, linear_B)
            model_fusion.encoder.layers[i].ffn.linear1 = linear_fusion

            # layer 2
            linear_A = modelA.encoder.layers[i].ffn.linear2
            linear_B = modelB.encoder.layers[i].ffn.linear2

            linear_fusion = linear_averaging(linear_A, linear_B)
            model_fusion.encoder.layers[i].ffn.linear2 = linear_fusion

            # iv) layer norm 2
            norm_gamma_A = modelA.encoder.layers[i].norm2.gamma
            norm_gamma_B = modelB.encoder.layers[i].norm2.gamma

            norm_gamma_fusion = weight_averaging(norm_gamma_A, norm_gamma_B)
            model_fusion.encoder.layers[i].norm2.gamma = torch.nn.Parameter(norm_gamma_fusion)

            norm_beta_A = modelA.encoder.layers[i].norm2.beta
            norm_beta_B = modelB.encoder.layers[i].norm2.beta

            norm_beta_fusion = weight_averaging(norm_beta_A, norm_beta_B)
            model_fusion.encoder.layers[i].norm2.beta = torch.nn.Parameter(norm_beta_fusion)

        # 2) MLP head
        linear_A = modelA.linear
        linear_B = modelB.linear

        linear_fusion = linear_averaging(linear_A, linear_B)
        model_fusion.linear = linear_fusion

    print('fusion successful')
    return model_fusion.to(device)


def getSupport(model, trainloader, l, alignment="acts", numOfBatches=10):
    '''
    Get the support matrices using Activation-based ("acts") or Weight-based ("wts") alignment
    '''
    if alignment == "acts":
        activation = None
        for i, data in enumerate(trainloader, 0):
            if i >= numOfBatches:
                break

            inputs, targets = data
            outputs = model(inputs)

            if activation is None:
                activation = model.actMatrix[l]
            else:
                activation = torch.cat((activation, model.actMatrix[l]))

        return activation

    elif alignment == "wts":
        return model.state_dict()[l]


def fusion(modelA, modelB, weights_nameA, weights_nameB, weightsA, weightsB, transport_matrix, beta, train_iter):
    """Aligns the parameters of model A and B and fuses them using OT"""
    # Put parent models on cpu (if not already)
    modelA = modelA.cpu()
    modelB = modelB.cpu()

    # Get Y-support (model B)
    support_y = getSupport(modelB, train_iter, weights_nameB, alignment="wts")

    # Align the weights from the first model
    aligned_weights = torch.matmul(weightsA, torch.matmul(transport_matrix, torch.diag(1 / beta)))

    # Get X-support (model A)
    n = weightsA.shape[0]
    alpha = torch.ones(n) * (1 / n)
    support_x = getSupport(modelA, train_iter, weights_nameA, alignment="wts")

    # Calculate the euclidean distance between the supports
    distance = ot.dist(support_x, support_y)

    # Calculate beta
    m = weightsB.shape[0]
    beta = torch.ones(m) * (1 / m)

    # Calculate the transport matrix using optimal transport
    transport_matrix = torch.from_numpy(ot.emd(alpha.numpy(), beta.numpy(), distance.detach().numpy())).float().reshape(
        (n, m))

    # Align model neurons
    aligned_model = torch.matmul(torch.diag(1 / beta), torch.matmul(transport_matrix.T, aligned_weights))

    # Get the weights at layer "idx" from the second model
    fused = (aligned_model + weightsB) / 2

    return fused, transport_matrix, beta


def fusion_multihead(modelA, modelB, nameA, nameB, weightA, weightB, transport_matrix, beta, train_iter):
    head = modelA.n_head

    support_y = weightB
    support_x = weightA
    # Get the weights at layer "idx" from the first model
    W_A = weightA
    W_B = weightB
    # Initialize the fused model and transport matrix
    fused = torch.empty(W_B.shape)
    transport_matrix_new = torch.zeros((weightA.shape[0], weightB.shape[0]))
    stride = weightB.shape[0] // head
    for i in range(0, weightB.shape[0], stride):
        # Align the weights from the first model
        aligned_W = torch.matmul(W_A[i:i+stride, :], torch.matmul(transport_matrix, torch.diag(1 / beta)))
        # Get the X-Support
        n = W_A.shape[0] // head
        alpha = torch.ones(n) * (1/n)
        # Calculate the euclidean distance between the supports
        distance = ot.dist(support_x[i:i+stride, :], support_y[i:i+stride, :])
        # Calculate beta
        m = W_B.shape[0] // head
        beta_new = torch.ones(m) * (1/m)
        # Calculate the transport matrix using optimal transport
        transport_matrix_new[i:i+stride, i:i+stride] = torch.from_numpy(ot.emd(alpha.numpy(), beta_new.numpy(), distance.detach().numpy())).float().reshape((n, m))
        # Align model neurons
        aligned_model = torch.matmul(torch.diag(1 / beta_new), torch.matmul(transport_matrix_new[i:i+stride, i:i+stride].T, aligned_W))
        # Get the weights at layer "idx" from the second model
        fused[i:i+stride, :] = (aligned_model + W_B[i:i+stride, :]) / 2
    return fused, transport_matrix_new, beta_new


def fusion_multilayer(modelA, modelB, nameA, nameB, weightA, weightB, transport_matrix, beta, train_iter):
    layers = len(modelA.encoder.layers)
    layer = int(nameA.partition("layers.")[2][0])

    W_A = torch.empty((weightA.shape[0] * layers, weightA.shape[1]))
    W_B = torch.empty((weightB.shape[0] * layers, weightB.shape[1]))

    for l in range(layers):
        W_A[l * weightB.shape[0] : l * weightB.shape[0] + weightB.shape[0], :] = modelA.state_dict()[nameA.replace(f"layers.{layer}", f"layers.{l}")]
        W_B[l * weightB.shape[0] : l * weightB.shape[0] + weightB.shape[0], :] = modelB.state_dict()[nameA.replace(f"layers.{layer}", f"layers.{l}")]

    support_y = W_B
    support_x = W_A

    # Initialize the fused model and transport matrix
    fused = torch.empty(W_B.shape)
    transport_matrix_new = torch.zeros((W_A.shape[0], W_B.shape[0]))
    # Align the weights from the first model
    aligned_W = torch.matmul(W_A, torch.matmul(transport_matrix, torch.diag(1 / beta)))
    # Get the X-Support
    n = W_A.shape[0]
    alpha = torch.ones(n) * (1/n)
    # Calculate the euclidean distance between the supports
    distance = ot.dist(support_x, support_y)
    # Calculate beta
    m = W_B.shape[0]
    beta = torch.ones(m) * (1/m)
    # Calculate the transport matrix using optimal transport
    transport_matrix = torch.from_numpy(ot.emd(alpha.numpy(), beta.numpy(), distance.detach().numpy())).float().reshape((n, m))
    # Align model neurons
    aligned_model = torch.matmul(torch.diag(1 / beta), torch.matmul(transport_matrix.T, aligned_W))
    # Get the weights at layer "idx" from the second model
    fused = (aligned_model + W_B) / 2
    dim = layer * weightB.shape[0]
    dim_to = layer * weightB.shape[0] + weightB.shape[0]
    return fused[dim:dim_to, :], transport_matrix_new[dim:dim_to, dim:dim_to], beta


def fusion_crossmultihead(modelA, modelB, nameA, nameB, weightA, weightB, transport_matrix, beta, train_iter):
    head = modelA.n_head

    W_A_head = weightA.view(head, -1)
    W_B_head = weightB.view(head, -1)

    m = W_B_head.shape[1]
    beta_head = torch.ones(m) * (1/m)
    transport_matrix_head = torch.matmul(torch.diag(beta_head), torch.eye(m))

    support_y = getSupport(modelB, train_iter, nameB, alignment="wts")
    support_x = getSupport(modelA, train_iter, nameA, alignment="wts")

    aligned_W = torch.matmul(W_A_head, torch.matmul(transport_matrix_head, torch.diag(1 / beta_head)))

    dist_head = ot.dist(support_x.view(head, -1), support_y.view(head, -1))

    n = W_A_head.shape[0]
    alpha_head = torch.ones(n) * (1/n)

    m = W_B_head.shape[0]
    beta_head = torch.ones(m) * (1/m)

    transport_matrix_new = torch.from_numpy(ot.emd(alpha_head.numpy(), beta_head.numpy(), dist_head.detach().numpy())).float().reshape((n, m))

    aligned_W_A = torch.matmul(torch.diag(1 / beta_head), torch.matmul(transport_matrix_new.T, aligned_W))
    aligned_W_A = aligned_W_A.view(weightA.shape)

    return fusion_multihead(modelA, modelB, nameA, nameB, aligned_W_A, weightB, transport_matrix, beta, train_iter)


def fusion_multihead_multilayer(modelA, modelB, nameA, nameB, weightA, weightB, transport_matrix, beta, train_iter):
    layers = len(modelA.encoder.layers)
    heads = modelA.n_head
    layer = int(nameA.partition("layers.")[2][0])

    W_A = torch.empty((weightA.shape[0] * layers, weightA.shape[1]))
    W_B = torch.empty((weightB.shape[0] * layers, weightB.shape[1]))

    stride = weightB.shape[0] // heads
    for l in range(layers):
        W_A[l * weightB.shape[0] : l * weightB.shape[0] + weightB.shape[0], :] = modelA.state_dict()[nameA.replace(f"layers.{layer}", f"layers.{l}")]
        W_B[l * weightB.shape[0] : l * weightB.shape[0] + weightB.shape[0], :] = modelB.state_dict()[nameA.replace(f"layers.{layer}", f"layers.{l}")]

    support_y = W_B
    support_x = W_A

    # Initialize the fused model and transport matrix
    fused = torch.empty(W_B.shape)
    transport_matrix_new = torch.zeros((W_A.shape[0], W_B.shape[0]))
    stride = W_B.shape[0] // (heads * layers)
    for i in range(0, W_B.shape[0], stride):
        # Align the weights from the first model
        aligned_W = torch.matmul(W_A[i:i+stride, :], torch.matmul(transport_matrix, torch.diag(1 / beta)))
        # Get the X-Support
        n = W_A.shape[0] // (heads * layers)
        alpha = torch.ones(n) * (1/n)
        # Calculate the euclidean distance between the supports
        distance = ot.dist(support_x[i:i+stride, :], support_y[i:i+stride, :])
        # Calculate beta
        m = W_B.shape[0] // (heads * layers)
        beta_new = torch.ones(m) * (1/m)
        # Calculate the transport matrix using optimal transport
        transport_matrix_new[i:i+stride, i:i+stride] = torch.from_numpy(ot.emd(alpha.numpy(), beta_new.numpy(), distance.detach().numpy())).float().reshape((n, m))
        # Align model neurons
        aligned_model = torch.matmul(torch.diag(1 / beta_new), torch.matmul(transport_matrix_new[i:i+stride, i:i+stride].T, aligned_W))
        # Get the weights at layer "idx" from the second model
        fused[i:i+stride, :] = (aligned_model + W_B[i:i+stride, :]) / 2
    dim = layer * weightB.shape[0]
    dim_to = layer * weightB.shape[0] + weightB.shape[0]
    return fused[dim:dim_to, :], transport_matrix_new[dim:dim_to, dim:dim_to], beta_new


# different mass distribution for fusion
def uniform_mass(n_support: int):
    '''
    Input
        n_support (int): number of supports
    '''
    return torch.ones(n_support) * (1 / n_support)

def random_mass(n_support: int):
    '''
    Input
        n_support (int): number of supports
    '''
    action_logits = torch.rand(n_support)
    action_probs = nn.functional.softmax(action_logits, dim=-1)
    return action_probs

def gaussian_mass(n_support: int):
    '''
    Normal distribution with mean 0 and std 1
    Input
        n_support (int): number of supports
    '''
    action_logits = torch.empty(n_support)
    action_probs = nn.functional.softmax(action_logits.normal_(), dim=-1)
    return action_probs

def geometric_mass(n_support: int):
    '''
    Geometric distribution with success probability 0.5
    Input
        n_support (int): number of supports
    '''
    action_logits = torch.empty(n_support)
    action_probs = nn.functional.softmax(action_logits.geometric_(0.5), dim=-1)
    return action_probs


def ot_fusion(modelA, modelB, train_iter, embedding, pad_idx, voc_size, device, fusion_ratio=0.5, drop_prob=0.5, variation='standard', pmd_name='uniform'):
    """Fuses models A, B together using optimal transport"""
    # Initialize new model
    model_fusion = new_model(embedding, pad_idx, voc_size, device, drop_prob=drop_prob, n_head=modelA.n_head) # init model
    a = fusion_ratio

    # Initialize fused weights dictionary
    W_fusion = dict.fromkeys(list(modelA.state_dict().keys()))

    # Initialize the algorithm
    m = list(modelB.state_dict().items())[1][1].shape[1]
    probability_mass_distributions = lambda name: {'uniform': uniform_mass,
                                                   'random': random_mass,
                                                   'gaussian': gaussian_mass,
                                                   'geometric': geometric_mass}[name]
    beta = probability_mass_distributions(pmd_name)(m)
    transport_matrix = torch.matmul(torch.diag(beta), torch.eye(m))

    # Fusion via Optimal Transport
    for (nameA, weightA), (nameB, weightB) in zip(modelA.named_parameters(), modelB.named_parameters()):
        weightA = weightA.to('cpu')
        weightB = weightB.to('cpu')
        if nameA == "encoder.emb.tok_emb.embedding.weight":
            W_fusion[nameA] = weightA  # all models use same embedding
        else:
            if "weight" in nameA:
                if "encoder" in nameA:
                    if "concat" not in nameA and "linear" not in nameA:
                        # select fusion function
                        fusion_variant = lambda var: {
                            'standard': fusion,
                            'multihead': fusion_multihead,
                            'cross-multihead': fusion_crossmultihead,
                            'multilayer': fusion_multilayer,
                            'multihead-multilayer': fusion_multihead_multilayer
                        }[var]
                        W_fusion[nameA], transport_matrix_triplet, _ = fusion_variant(variation)(modelA,
                                                                                                 modelB,
                                                                                                 nameA,
                                                                                                 nameB,
                                                                                                 weightA,
                                                                                                 weightB,
                                                                                                 transport_matrix,
                                                                                                 beta,
                                                                                                 train_iter)
                    else:
                        W_fusion[nameA], transport_matrix, beta = fusion(modelA, modelB, nameA, nameB, weightA, weightB,
                                                                         transport_matrix, beta, train_iter)
                else:
                    W_fusion[nameA] = a * weightA + (1 - a) * weightB
            elif "bias" in nameA:
                if "encoder" in nameA:
                    if "concat" not in nameA and "linear" not in nameA:
                        m = weightB.shape[0]
                        beta_bias = torch.ones(m) * (1 / m)

                        W_A_bias = weightA.reshape(m, 1)

                        aligned_bias = torch.matmul(torch.diag(1 / beta_bias),
                                                    torch.matmul(transport_matrix_triplet.T, W_A_bias))
                        aligned_bias = aligned_bias.reshape(m)

                        W_fusion[nameA] = (aligned_bias + weightB) / 2
                    else:
                        m = weightB.shape[0]
                        beta_bias = torch.ones(m) * (1 / m)

                        W_A_bias = weightA.reshape(m, 1)

                        aligned_bias = torch.matmul(torch.diag(1 / beta_bias),
                                                    torch.matmul(transport_matrix.T, W_A_bias))
                        aligned_bias = aligned_bias.reshape(m)

                        W_fusion[nameA] = (aligned_bias + weightB) / 2
                else:
                    W_fusion[nameA] = a * weightA + (1 - a) * weightB
            else:
                W_fusion[nameA] = a * weightA + (1 - a) * weightB

    # Assign the weights
    with torch.no_grad():
        for name, param in model_fusion.named_parameters():
            param.data = torch.nn.Parameter(W_fusion[name])

    print('fusion successful')
    return model_fusion


def test_fusion(modelA, modelB, model_fusion, test_iter, device):
    # ensure all models on same device
    modelA = modelA.to(device)
    modelB = modelB.to(device)
    model_fusion = model_fusion.to(device)

    # test fusion
    test_loss_A, test_acc_A = validation(modelA, test_iter, None, nn.CrossEntropyLoss(), device)
    test_loss_B, test_acc_B = validation(modelB, test_iter, None, nn.CrossEntropyLoss(), device)
    test_loss_fusion, test_acc_fusion = validation(model_fusion, test_iter, None, nn.CrossEntropyLoss(), device)

    # put into cpu
    test_loss_A = test_loss_A.to('cpu') if isinstance(test_loss_A, torch.Tensor) else test_loss_A
    test_acc_A = test_acc_A.to('cpu') if isinstance(test_acc_A, torch.Tensor) else test_acc_A

    test_loss_B = test_loss_B.to('cpu') if isinstance(test_loss_B, torch.Tensor) else test_loss_B
    test_acc_B = test_acc_B.to('cpu') if isinstance(test_acc_B, torch.Tensor) else test_acc_B

    test_loss_fusion = test_loss_fusion.to('cpu') if isinstance(test_loss_fusion, torch.Tensor) else test_loss_fusion
    test_acc_fusion = test_acc_fusion.to('cpu') if isinstance(test_acc_fusion, torch.Tensor) else test_acc_fusion

    # visualize
    fig, ax = plt.subplots()

    metrics_A = [test_loss_A, test_acc_A]
    metrics_B = [test_loss_B, test_acc_B]
    metrics_fusion = [test_loss_fusion, test_acc_fusion]
    metrics = ['loss', 'accuracy']
    x = np.arange(len(metrics))  # positions of bars (1 per metric)
    width = 0.25  # the width of the bars

    # bars
    rects1 = ax.bar(x - width, metrics_A, width, label='model A')
    rects2 = ax.bar(x, metrics_B, width, label='model B')
    rects3 = ax.bar(x + width, metrics_fusion, width, label='model fusion')

    # number on top of bars
    for rect in (-1, 0, 1):  # for each bar
        metric = [metrics_A, metrics_B, metrics_fusion][rect + 1]  # choose values for model A, B or fusion
        for i, x_ in enumerate(x + rect * width):  # x_ is position of bar
            y_ = metric[i] if not isinstance(metric[i], torch.Tensor) else metric[i].numpy()
            plt.text(x=x_ - 0.3 * width, y=y_ + 0.03, s=np.round(y_, 3))

    ax.set_ylabel('Score')
    ax.set_title('Test metrics by models')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.ylim([0, max([max(element) for element in [metrics_A, metrics_B, metrics_fusion]]) + 0.2])

    plt.show()


# wrapper function to optimize weighting factor
def weighted_fusion(modelA, modelB, train_iter, valid_iter, embedding, pad_idx, voc_size, device, variation='standard', pmd_name='uniform'):
    def objective(trial):
        weighting_factor = trial.suggest_float('weighting_factor', 0, 1)

        # weighted fusion
        model_fusion = ot_fusion(modelA, modelB, train_iter, embedding, pad_idx, voc_size, device, fusion_ratio=weighting_factor, variation=variation, pmd_name=pmd_name)
        model_fusion.to(device)

        # validate fusion model
        criterion = torch.nn.CrossEntropyLoss()
        val_loss, val_acc, f1 = validation(model_fusion, valid_iter, criterion, device)

        return val_loss
    return objective


def calibrate_norm_layer(fusedModel, train_iter, device, learning_rate=0.01):
    optimizer = torch.optim.Adam(fusedModel.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    clip = 1

    fusedModel.train()
    for name, param in fusedModel.named_parameters():
        if "weight" in name or "bias" in name:
            param.requires_grad = False

    for i, batch in enumerate(tqdm(train_iter)):
        src = batch[0] # X
        trg = batch[1] # y
        src, trg = torch.tensor(src).to(device), torch.tensor(trg).to(device) # put to cpu/gpu
        optimizer.zero_grad() # reset optimizer
        output = fusedModel(src) # predict
        y_pred = torch.argmax(output, dim=-1) # logits -> labels
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg.to(torch.int64)
        loss = criterion(output_reshape, trg) # calculate loss
        agreements = torch.eq(y_pred, trg)
        accuracy = torch.mean(agreements.double()) # calculate accuracy
        loss.backward() # backward pass

        torch.nn.utils.clip_grad_norm_(fusedModel.parameters(), clip)
        optimizer.step() # optimize model

    for name, param in fusedModel.named_parameters():
        param.requires_grad = True


def train_last_layer(fusedModel, train_iter, device, learning_rate=1e-3):
    optimizer = torch.optim.Adam(fusedModel.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    clip = 1

    fusedModel.train()
    for name, param in fusedModel.named_parameters():
        if "encoder" in name:
            param.requires_grad = False

    for i, batch in enumerate(tqdm(train_iter)):
        src = batch[0] # X
        trg = batch[1] # y
        src, trg = torch.tensor(src).to(device), torch.tensor(trg).to(device) # put to cpu/gpu
        optimizer.zero_grad() # reset optimizer
        output = fusedModel(src) # predict
        y_pred = torch.argmax(output, dim=-1) # logits -> labels
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg.to(torch.int64)
        loss = criterion(output_reshape, trg) # calculate loss
        agreements = torch.eq(y_pred, trg)
        accuracy = torch.mean(agreements.double()) # calculate accuracy
        loss.backward() # backward pass

        torch.nn.utils.clip_grad_norm_(fusedModel.parameters(), clip)
        optimizer.step() # optimize model


# TEST FUNCTIONS
def scores_to_latex(scores, model_names):
    df = pd.DataFrame.from_dict(scores)

    # for visualization
    temp = pd.DataFrame.from_dict(scores)
    n_rows, n_cols = temp.shape
    for row in range(n_rows):
        for col in range(n_cols):
            entry = temp.iloc[row, col]
            temp.iloc[row, col] = [np.round(element, 3) if not isinstance(element, torch.Tensor) else np.round(element.numpy(), 3) for element in entry]
    print(temp)

    # entries like: mean +- std
    n_rows, n_cols = df.shape
    for row in range(n_rows):
        df.iloc[row, 0] = str(np.round(np.mean(df.iloc[row, 0]), 3)) + ' ± ' + str(np.round(np.std(df.iloc[row, 0]), 3)) # loss
        df.iloc[row, 1] = str(np.round(np.mean(df.iloc[row, 1]), 3)) + ' ± ' + str(np.round(np.std(df.iloc[row, 1]), 3)) # accuracy
        df.iloc[row, 2] = str(np.round(np.mean(df.iloc[row, 2]), 3)) + ' ± ' + str(np.round(np.std(df.iloc[row, 2]), 3)) # f1

    # rename rows, cols
    df.columns = ['Loss', 'Accuracy', 'F1 score']
    df.index = model_names

    # boldify highest score
    for col in (0, 1, 2):
        if col == 0:
            index_max = np.argmin([float(entry.split('±')[0]) for entry in df.iloc[:, col]])
        else:
            index_max = np.argmax([float(entry.split('±')[0]) for entry in df.iloc[:, col]])
        entry = df.iloc[index_max, col]
        entry = 'BOLD{' + entry + '}'
        df.iloc[index_max, col] = entry


    # convert to latex
    latex = df.to_latex(index=True,
                        bold_rows=True,
                        caption='Model performance (5-fold CV)',
                        position='H').replace('BOLD\\', r'\textbf').replace('\}', '}')

    # print as latex
    print(latex)

    return latex
