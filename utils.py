import torch
import torch.nn as nn
import math
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchtext.data import Field, BucketIterator
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
    data = pd.read_csv("./Data/IMDB Dataset.csv", encoding='ISO-8859-1')

    # convert string label to binary (int) label (spam: 1, non-spam: 0)
    data["sentiment"] = data['sentiment'].apply(lambda x: int(x == "positive"))

    return data


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
                                  max_len=256,
                                  d_model=16,
                                  ffn_hidden=32,
                                  n_head=1,
                                  n_layers=1,
                                  drop_prob=0.5,
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


def validation(model, iterator, optimizer, criterion, device):
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
    if epoch == 'unrestricted':
        epoch = int(1e6) # large enough
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
            val_loss, val_acc = validation(model, valid_iter, optimizer, criterion, device)

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
    for e in range(epoch):
        # loss, metrics for current epoch
        epoch_loss = 0
        epoch_acc = 0

        # batches
        for i, batch in enumerate(tqdm(iterator)):
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
        val_loss, val_acc = validation(model, valid_iter, optimizer, criterion, device)

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
        print(f"Epoch: {e + 1}  Train Loss: {epoch_loss / len(iterator):.4f} \
              Validation Loss: {val_loss:.4f} \
              Train acc: {epoch_acc:.4f}, \
              Val acc: {val_acc:.4f}, \
              Learning Rate : {optimizer.param_groups[0]['lr'] :.4f}")

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
        linear_averaged.weight, linear_averaged.bias = torch.nn.Parameter(weight_averaging(*weights)), \
                                                       torch.nn.Parameter(weight_averaging(*biases))

    return linear_averaged


def vanilla_fusion(modelA, modelB, pad_idx, voc_size, embedding, device):
    # init
    model_fusion = new_model(embedding, pad_idx, voc_size, device)  # init model

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
            # TODO: correct?
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
            # TODO: correct?
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


def ot_fusion(modelA, modelB, train_iter, embedding, pad_idx, voc_size, device, fusion_ratio=0.5):
    """Fuses models A, B together using optimal transport"""
    # Initialize new model
    model_fusion = new_model(embedding, pad_idx, voc_size, device)  # init model
    a = fusion_ratio

    # Initialize fused weights dictionary
    W_fusion = dict.fromkeys(list(modelA.state_dict().keys()))

    # Initialize the algorithm
    m = list(modelB.state_dict().items())[1][1].shape[1]
    beta = torch.ones(m) * (1 / m)
    transport_matrix = torch.matmul(torch.diag(beta), torch.eye(m))

    # Fusion via Optimal Transport
    for (nameA, weightA), (nameB, weightB) in zip(modelA.named_parameters(), modelB.named_parameters()):
        if nameA == "encoder.emb.tok_emb.embedding.weight":
            W_fusion[nameA] = weightA  # all models use same embedding
        else:
            if "weight" in nameA:
                if "encoder" in nameA:
                    if "concat" not in nameA and "linear" not in nameA:
                        # query, key and value matrices get the same transport matrix
                        W_fusion[nameA], transport_matrix_triplet, _ = fusion(modelA, modelB, nameA, nameB, weightA, weightB,
                                                                              transport_matrix, beta, train_iter)
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

                        aligned_bias = torch.matmul(torch.diag(1 / beta_bias), torch.matmul(transport_matrix_triplet.T, W_A_bias))
                        aligned_bias = aligned_bias.reshape(m)

                        W_fusion[nameA] = (aligned_bias + weightB) / 2
                    else:
                        m = weightB.shape[0]
                        beta_bias = torch.ones(m) * (1 / m)

                        W_A_bias = weightA.reshape(m, 1)

                        aligned_bias = torch.matmul(torch.diag(1 / beta_bias), torch.matmul(transport_matrix.T, W_A_bias))
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
