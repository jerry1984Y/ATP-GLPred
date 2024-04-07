import datetime
import math

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import torch
import torch.nn as nn

from sklearn import metrics

from LossFunction.focalLoss import FocalLoss_v2
from utility import read_file_list, save_prob_label, masked_softmax, read_file_list_from_seq_label


def read_data_file_trip(filename):
    f = open(filename)
    data = f.readlines()
    f.close()

    results=[]
    block=len(data)//2
    for index in range(block):
        item=data[index*2+0].split()
        name =item[0].strip()
        results.append(name)
    return results

def coll_paddding(batch_traindata):
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)
    feature_plms = []
    features_proteins = []
    train_y = []

    for data in batch_traindata:
        feature_plms.append(data[0])
        features_proteins.append(data[1])
        train_y.append(data[2])
    data_length = [len(data) for data in feature_plms]

    feature_plms = torch.nn.utils.rnn.pad_sequence(feature_plms, batch_first=True, padding_value=0)
    features_proteins = torch.nn.utils.rnn.pad_sequence(features_proteins, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)
    return feature_plms, features_proteins, train_y, torch.tensor(data_length)

class BioinformaticsDataset(Dataset):
    # X: list of filename
    def __init__(self, X):
        self.X = X

    def __getitem__(self, index):
        filename = self.X[index]
        # esm_embedding1280 prot_embedding  esm_embedding2560 msa_embedding
        #df0 = pd.read_csv('DataSet/prot_embedding/' + filename + '.data', header=None)
        df0 = pd.read_csv('DataSet/prot_embedding/' + filename + '.data', header=None)
        prot = df0.values.astype(float).tolist()
        prot = torch.tensor(prot)

        agv_pro = torch.mean(prot, dim=0)
        agv_pro = agv_pro.repeat(prot.shape[0], 1)

        agv_res = torch.mean(prot, dim=1)
        agv_res = agv_res.unsqueeze(dim=1)

        agv_pro = torch.cat((agv_pro, agv_res), dim=1)

        df2 = pd.read_csv('DataSet/prot_embedding/' + filename + '.label', header=None)
        label = df2.values.astype(int).tolist()
        label = torch.tensor(label)
        # reduce 2D to 1D
        label = torch.squeeze(label)
        # ADP-0; AMP-1; ATP-2; GDP-3; GTP-4


        return prot, agv_pro, label

    def __len__(self):
        return len(self.X)
class ATPModule(nn.Module):
    def __init__(self):
        super(ATPModule, self).__init__()
        #20-256-512-256-128-l512-l64-l2  3,5,3,5
        # 21 psfm
        # 30 hmm
        # 20 pssm
        # 13 chem
        #self.fc0=nn.Linear(1024,128)

        self.input_block = nn.Sequential(
            nn.LayerNorm(1024, eps=1e-6)
            , nn.Linear(1024, 128)
            , nn.LeakyReLU()
        )

        self.hidden_block = nn.Sequential(
            nn.LayerNorm(128, eps=1e-6)
            , nn.Dropout(0.2)
            , nn.Linear(128, 128)
            , nn.LeakyReLU()
            , nn.LayerNorm(128, eps=1e-6)
        )


        encoderlayer=nn.TransformerEncoderLayer(d_model=128,nhead=2,dim_feedforward=64*2,dropout=0.2,batch_first=True)
        self.encoder=nn.TransformerEncoder(encoder_layer=encoderlayer,num_layers=1)
        self.share_task_block1 = nn.Sequential(nn.Conv1d(1024, 512, 1, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(512, 256, 1, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(256, 128, 1, padding='same'),
                                               nn.ReLU(True))
        self.share_task_block2 = nn.Sequential(nn.Conv1d(1024, 512, 3, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(512, 256, 3, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(256, 128, 3, padding='same'),
                                               nn.ReLU(True))
        self.share_task_block3 = nn.Sequential(nn.Conv1d(1024, 512, 5, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(512, 256, 5, padding='same'),
                                               nn.ReLU(True),
                                               nn.Conv1d(256, 128, 5, padding='same'),
                                               nn.ReLU(True))
        # self.lstm = nn.LSTM(
        #     input_size=128,
        #     hidden_size=64,
        #     num_layers=2,
        #     bidirectional=True,
        #     batch_first=True,
        #     dropout=0.5
        # )
        self.fc=nn.Sequential(nn.Linear(128,256),
                                          nn.Dropout(0.5),
                                          nn.Linear(256,64),
                                          nn.Dropout(0.5),
                                          nn.Linear(64,2))

    def create_src_lengths_mask(self, batch_size: int, src_lengths):
        max_src_len = int(src_lengths.max())
        src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
        src_indices = src_indices.expand(batch_size, max_src_len)
        src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
        # returns [batch_size, max_seq_len]
        return (src_indices < src_lengths).float().detach()

    def forward(self,prot0,f0agv,data_length):

        emb=self.input_block(prot0)
        emb=self.hidden_block(emb)
        mask = 1 - self.create_src_lengths_mask(emb.size(0), data_length)
        emb2 =self.encoder(emb, src_key_padding_mask=mask)

        # x = torch.nn.utils.rnn.pack_padded_sequence(emb, data_length.to('cpu'), batch_first=True)
        # x, (h_n, h_c) = self.lstm(x)
        # protlstm, output_lens = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        prot0=prot0.permute(0,2,1)
        p1=self.share_task_block1(prot0)
        p2=self.share_task_block2(prot0)
        p3=self.share_task_block3(prot0)
        #
        p1=p1.permute(0,2,1)
        p2=p2.permute(0,2,1)
        p3=p3.permute(0,2,1)
        p=p1+p2+p3+emb2

        predict=self.fc(p)
        return predict

def train(itrainfile,modelstoreapl):
    file = read_data_file_trip(itrainfile)
    train_set = BioinformaticsDataset(file)
    model = ATPModule()
    epochs =30

    model = model.to(device)
    train_loader = DataLoader(dataset=train_set, batch_size=16,shuffle=True, num_workers=16,  pin_memory=True, persistent_workers=True,
                              collate_fn=coll_paddding)
    best_val_loss = 3000

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    per_cls_weights = torch.FloatTensor([0.2,0.8]).to(device)
    fcloss=FocalLoss_v2(alpha=per_cls_weights, gamma=2)
    model.train()
    for i in range(epochs):

        epoch_loss_train = 0.0
        nb_train = 0
        for prot_x,f0agv,data_y,length in train_loader:

            y_pred = model(prot_x.to(device),f0agv.to(device),length.to(device))
            y_pred = torch.nn.utils.rnn.pack_padded_sequence(y_pred, length.to('cpu'), batch_first=True)
            data_y = torch.nn.utils.rnn.pack_padded_sequence(data_y, length, batch_first=True)
            data_y = data_y.to(device)

            single_loss=fcloss(y_pred.data,data_y.data)
            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()
            epoch_loss_train = epoch_loss_train + single_loss.item()
            nb_train+=1
        epoch_loss_avg=epoch_loss_train/nb_train
        if best_val_loss > epoch_loss_avg:
            model_fn = modelstoreapl
            torch.save(model.state_dict(), model_fn)
            best_val_loss = epoch_loss_avg
            if i % 10 == 0:
                print('epochs ', i)
                print("Save model, best_val_loss: ", best_val_loss)
def test(itestfile,modelstoreapl):
    file = read_data_file_trip(itestfile)
    test_set = BioinformaticsDataset(file)

    test_load = DataLoader(dataset=test_set, batch_size=32,
                           num_workers=16, pin_memory=True, persistent_workers=True, collate_fn=coll_paddding)
    model = ATPModule()
    model = model.to(device)

    print("==========================Test RESULT================================")

    model.load_state_dict(torch.load(modelstoreapl))
    model.eval()

    arr_probs = []
    arr_labels = []
    arr_labels_hyps = []
    with torch.no_grad():
        for prot_x,f0agv,data_y, length in test_load:
            y_pred = model(prot_x.to(device),f0agv.to(device),length.to(device))
            y_pred = torch.nn.utils.rnn.pack_padded_sequence(y_pred, length.to('cpu'), batch_first=True)
            y_pred=y_pred.data
            y_pred=torch.nn.functional.softmax(y_pred,dim=1)
            arr_probs.extend(y_pred[:, 1].to('cpu'))
            y_pred=torch.argmax(y_pred, dim=1).to('cpu')
            data_y = torch.nn.utils.rnn.pack_padded_sequence(data_y, length, batch_first=True)
            arr_labels.extend(data_y.data)
            arr_labels_hyps.extend(y_pred)

    print('-------------->')

    auc =metrics.roc_auc_score(arr_labels, arr_probs)
    precision_1, recall_1, threshold_1 = metrics.precision_recall_curve(arr_labels, arr_probs)
    aupr_1 = metrics.auc(recall_1, precision_1)
    print('acc ', metrics.accuracy_score(arr_labels, arr_labels_hyps))
    print('balanced_accuracy ', metrics.balanced_accuracy_score(arr_labels, arr_labels_hyps))
    tn, fp, fn, tp = metrics.confusion_matrix(arr_labels, arr_labels_hyps).ravel()
    print('tn, fp, fn, tp ',tn, fp, fn, tp )
    print('MCC ', metrics.matthews_corrcoef(arr_labels, arr_labels_hyps))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1score = 2 * tp / (2 * tp + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    youden = sensitivity + specificity - 1
    print('sensitivity ', sensitivity)
    print('specificity ', specificity)
    print('precision ', precision)
    print('recall ', recall)
    print('f1score ', f1score)
    print('youden ', youden)
    print('auc', auc)
    print('AUPR ', aupr_1)
    print('<----------------')
    save_prob_label(arr_probs, arr_labels, modelstoreapl+'.csv')
    print('<----------------save to csv finish')


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    cuda = torch.cuda.is_available()
    torch.cuda.set_device(1)
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    # train_file_list = read_file_list_from_seq_label('DataSet/atp-388.txt')
    # test_file_list = read_file_list_from_seq_label('DataSet/atp-41-for-388.txt')
    train_taskfile = 'DataSet/atp-388.txt'  # ALL_IRON_Train.txt
    test_taskfie = 'DataSet/atp-41-for-388.txt'  # ALL_IRON_Test.txt
    # train_taskfile = 'DataSet/atp-227.txt'  # ALL_IRON_Train.txt
    # test_taskfie = 'DataSet/atp-17-for-227.txt'  # ALL_IRON_Test.txt

    a = str(datetime.datetime.now())
    a = a.replace(':', '_')
    storeapl = 'ATP-Multi/ATP288_MsCNN' + '_' + a + '.pkl'
    train(train_taskfile,storeapl)
    test(test_taskfie,storeapl)
    #test(test_taskfie,'ATP-Multi/ATP__2024-01-20 20_23_00.791050.pkl')

