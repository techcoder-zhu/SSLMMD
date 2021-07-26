import pandas as pd
import random
import math
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model_lstm import BatchProgramClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels, feature = [], [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        labels.append(item[2]) # item[2]-1
        feature.append(item['Traditional'])
    return data, torch.LongTensor(labels), torch.Tensor(feature)


def run(src_pro,dst):
    pd.set_option('display.max_columns', 1000)  # 显示全部列
    pd.set_option('display.max_row', 5000)  # 显示全部行
    pd.set_option('display.width', 1000)  # 设置数据的显示长度（解决自动换行）

    root = 'data/'
    train_data = pd.read_pickle(root+'train/blocks.pkl')
    # print([column for column in train_data])
    # print(train_data)
    val_data = pd.read_pickle(root + 'dev/blocks.pkl')
    test_data = pd.read_pickle(root+'test/blocks.pkl')
    word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 2    # 104
    EPOCHS = 15
    BATCH_SIZE = 32   #64
    USE_GPU = False
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    print('Start training...')
    # training procedure
    best_model = model
    for epoch in range(EPOCHS):
        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        pred_train = []
        y_train = []
        i = 0
        j = 0
        while i < len(train_data)-((len(train_data)) % BATCH_SIZE):
            source_batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_labels, train_feature = source_batch
            if USE_GPU:
                train_inputs, train_labels,train_feature = train_inputs, train_labels,train_feature.cuda()

            target_batch = get_batch(test_data, j ,BATCH_SIZE)
            j += len(train_inputs)
            target_inputs,target_labels,target_feature = target_batch
            if len(target_inputs) < len(train_inputs):
                j = 0
                target_batch = get_batch(test_data, j, BATCH_SIZE)
                j += len(train_inputs)
                target_inputs, target_labels, target_feature = target_batch
            if USE_GPU:
                target_inputs, target_labels, target_feature = target_inputs,target_labels,target_feature.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output,mmd_loss = model(train_inputs,train_feature,train_labels,target_inputs, target_feature, target_labels)

            loss = loss_function(output, Variable(train_labels))
            loss += mmd_loss
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            pred_train += predicted
            y_train += train_labels
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item() * len(train_inputs)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        j = 0
        pred_val = []
        y_val = []
        while i < len(val_data)-((len(val_data)) % BATCH_SIZE):
            batch = get_batch(val_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            val_inputs, val_labels,val_feature = batch
            if USE_GPU:
                val_inputs, val_labels,val_feature = val_inputs, val_labels,val_feature.cuda()

            target_batch = get_batch(test_data, j, BATCH_SIZE)
            j += len(val_inputs)
            target_inputs, target_labels, target_feature = target_batch
            if len(target_inputs) < len(val_inputs):
                j = 0
                target_batch = get_batch(test_data, j, BATCH_SIZE)
                j += len(val_inputs)
                target_inputs, target_labels, target_feature = target_batch
            if USE_GPU:
                target_inputs, target_labels, target_feature = target_inputs, target_labels, target_feature.cuda()


            model.batch_size = len(val_labels)
            model.hidden = model.init_hidden()
            output,mmd_loss = model(val_inputs,val_feature,val_labels,target_inputs, target_feature, target_labels)

            loss = loss_function(output, Variable(val_labels))
            loss += mmd_loss
            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            pred_val += predicted
            y_val += val_labels
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item() * len(val_inputs)
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc/total > best_acc:
            best_model = model
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))

        print('Training Auc: ' + str(roc_auc_score(y_train, pred_train)) +
              ' Training F1: ' + str(f1_score(y_train, pred_train)) +
              ' Training recall: ' + str(recall_score(y_train, pred_train)) +
              ' Training precision: ' + str(precision_score(y_train, pred_train)))
        print('Validation Auc: ' + str(roc_auc_score(y_val, pred_val)) +
              ' Validation F1: ' + str(f1_score(y_val, pred_val)) +
              ' Validation recall: ' + str(recall_score(y_val, pred_val)) +
              ' Validation precision: ' + str(precision_score(y_val, pred_val)))

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    y = []
    pred = []
    model = best_model
    while i < len(test_data)-((len(test_data)) % BATCH_SIZE):
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels,test_feature = batch
        if USE_GPU:
            test_inputs, test_labels,test_feature = test_inputs, test_labels,test_feature.cuda()



        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output,mmd_loss = model(test_inputs,test_feature,test_labels,test_inputs,test_feature,test_labels)

        loss = loss_function(output, Variable(test_labels))
        loss += mmd_loss
        _, predicted = torch.max(output.data, 1)
        pred += predicted
        y += test_labels
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print('training on ' + src_pro + ' and test on ' + dst + '...')
    print("Testing results(Acc):", total_acc.item() / total)
    print('Testing Auc: ' + str(roc_auc_score(y, pred)))
    print(pred)
    pred = pred
    y_test = test_data['label']
    # prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average='binary')  # at threshold = 0.5
    # tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    # #     rec = tp/(tp+fn)
    #
    # FAR = fp / (fp + tn)  # false alarm rate
    # dist_heaven = math.sqrt((pow(1 - rec, 2) + pow(0 - FAR, 2)) / 2.0)  # distance to heaven

    # AUC = roc_auc_score(y, pred)
    #
    # result_df['defect_density'] = result_df['defective_commit_prob'] / result_df['LOC']  # predicted defect density
    # result_df['actual_defect_density'] = result_df['label'] / result_df['LOC']  # defect density
    #
    # result_df = result_df.sort_values(by='defect_density', ascending=False)
    # actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
    # actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)
    #
    test_data['cum_LOC'] = test_data['loc'].cumsum()
    # actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
    # actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()
    #
    real_buggy_commits = test_data[test_data['label'] == 1]
    #
    # label_list = list(result_df['label'])
    #
    # all_rows = len(label_list)

    # find Recall@20%Effort
    cum_LOC_20_percent = 0.2 * test_data.iloc[-1]['cum_LOC']
    buggy_line_20_percent = test_data[test_data['cum_LOC'] <= cum_LOC_20_percent]
    buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label'] == 1]
    recall_20_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))
    print(recall_20_percent_effort)

    # find Effort@20%Recall
    buggy_20_percent = real_buggy_commits.head(math.ceil(0.2 * len(real_buggy_commits)))
    buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
    effort_at_20_percent_LOC_recall = int(buggy_20_percent_LOC) / float(test_data.iloc[-1]['cum_LOC'])
    print(effort_at_20_percent_LOC_recall)