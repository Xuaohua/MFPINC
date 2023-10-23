import math
import re
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, auc, matthews_corrcoef, precision_score, recall_score, \
    confusion_matrix, roc_curve, roc_auc_score
import joblib
import random

from itertools import product
% config
Warning.simplerfilter = True

# device =torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:0')
device = torch.device('cpu')

seed_value = 42

torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ori_fea = pd.read_csv(r'../Features/All-features-of-the-training-set.csv')
ori_fea = ori_fea.drop(ori_fea.columns[0], axis=1)

label = [1] * 8000 + [0] * 8000

oy = pd.DataFrame(label, columns=['label'])


def extract_sequences_from_fasta(file_path):
    headers = []
    sequences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        current_header = ''
        current_sequence = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_header and current_sequence:
                    headers.append(current_header)
                    sequences.append(current_sequence)

                current_header = line[1:]
                current_sequence = ''
            else:
                current_sequence += line

        if current_header and current_sequence:
            headers.append(current_header)
            sequences.append(current_sequence)

    data = {'seqname': headers, 'seq': sequences}
    df = pd.DataFrame(data)
    return df


file_path = r'../Data/TestSets/cRNA/Gossypium darwinii.fasta'
datah = extract_sequences_from_fasta(file_path)
datah = datah.reset_index(drop=True)

print(datah.shape[0])

pattern = r'(?:^|\s)([^ ]+)'
datah['seqname'] = [re.search(pattern, identifier).group(1) for identifier in datah['seqname']]

data = datah

names = ['seqname', 'start', 'end', 'source ', 'accession', 'score ', 'startComplete ',
         'endComplete ', 'cdsCount ', 'cdsStarts ', 'cdsSizes ']
index_col = "seqname"

# The file path of the gene sequence to be tested
rna_cds = pd.read_csv(r'../Features/TestSets_cds_features/c_cds_test/Gossypium_darwinii', names=names, sep='\t')
rna_cds_label = rna_cds.shape[0]
print(rna_cds_label)
rna_cds = rna_cds.reset_index(drop=True)

datah_seqname = datah['seqname']
rna_cds_seqname = rna_cds['seqname']

indexes_to_drop = datah_seqname[~datah_seqname.isin(rna_cds_seqname)].index

datah.drop(indexes_to_drop, inplace=True)

label_data = [0] * (rna_cds_label) + [1] * (rna_cds_nc.shape[0])

y = pd.DataFrame(label_data, columns=['label'])
datah.reset_index(drop=True, inplace=True)

datah.insert(0, 'label', y)

data3 = datah.copy()


# compute k-mers features (k=1~4)
def Kmers_funct(seq, x):
    X = [None] * len(seq)
    for i in range(len(seq)):
        a = seq[i]
        t = 0
        l = []
        for index in range(len(a) - x + 1):
            t = a[index:index + x]
            if (len(t)) == x:
                l.append(t)
        X[i] = l
    return X


def nucleotide_type(k):
    z = []
    for i in product('ACGT', repeat=k):
        z.append(''.join(i))
    return z


def Kmers_frequency(seq, x):
    X = []
    char = nucleotide_type(x)
    for i in range(len(seq)):
        s = seq[i]
        frequence = []
        for a in char:
            number = s.count(a)
            char_frequence = number / (len(s) - x + 1)
            frequence.append(char_frequence)
        X.append(frequence)
    return X


feature_1mer = Kmers_funct(data.seq, 1)  # 1-mer
feature_2mer = Kmers_funct(data.seq, 2)  # 2-mer
feature_3mer = Kmers_funct(data.seq, 3)  # 3-mer

feature_1mer_frequency = Kmers_frequency(feature_1mer, 1)  # 1-mer
feature_2mer_frequency = Kmers_frequency(feature_2mer, 2)  # 2-mer
feature_3mer_frequency = Kmers_frequency(feature_3mer, 3)  # 3-mer

feature = pd.concat([pd.DataFrame(feature_1mer_frequency),
                     pd.DataFrame(feature_2mer_frequency),
                     pd.DataFrame(feature_3mer_frequency)], axis=1)
feature.columns = ['A', 'C', 'G', 'T', 'AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA',
                   'TC', 'TG', 'TT', 'AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT',
                   'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC',
                   'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT', 'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT',
                   'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC',
                   'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT']


def gc_content(dna_sequence):
    gc_percent = []
    for i in dna_sequence:
        gc_count = i.count('G') + i.count('C')
        total_count = len(i)
        gc_percent_eve = (gc_count / total_count) * 100
        gc_percent.append(gc_percent_eve)
    return gc_percent


gc_precent = gc_content(data.seq)
gc_precent_pd = pd.DataFrame(gc_precent)
gc_precent_pd.rename(columns={0: 'GC content'}, inplace=True)

feature.insert(0, 'GC content', gc_precent_pd)

feature = pd.DataFrame(feature)
feature.iloc[:, 0] = feature.iloc[:, 0] / 100

sequence_len = []
for i in range(data.shape[0]):
    sequence_len.append(len(data.seq[i]))
sequence_len = pd.DataFrame(sequence_len)
sequence_len.rename(columns={0: 'seq_len'}, inplace=True)

feature.insert(1, 'seq_len', sequence_len)
feature.rename(columns={'GC content': 'GC_con'}, inplace=True)

names = ['seqname', 'start', 'end', 'source ', 'accession', 'score ', 'startComplete ',
         'endComplete ', 'cdsCount ', 'cdsStarts ', 'cdsSizes ']
index_col = "seqname"

# The file path for the CDS features of the gene sequence to be tested
rna_cds = pd.read_csv(r'../Features/TestSets_cds_features/c_cds_test/Gossypium_darwinii', names=names, sep='\t',
                      index_col=index_col)
rna_cds = rna_cds.reset_index(drop=True)

data1 = data
data2 = data
for i, l in data2.iterrows():
    data1.iloc[i, 2] = len(l['seq'])

mylist = []
mylist = pd.DataFrame(mylist, columns=['score', 'cdsStarts', 'cdsStop', 'cdsSizes', 'cdsPercent'])

for index, row in rna_cds.iterrows():
    tem_pd_score = pd.DataFrame(row)
    mylist.loc[index, 'cdsStarts'] = row["start"]
    mylist.loc[index, 'cdsStop'] = row["end"]
    mylist.loc[index, 'cdsSizes'] = row["end"] - row["start"]
    mylist.loc[index, 'score'] = tem_pd_score.iloc[4, 0]
    mylist.loc[index, 'cdsPercent'] = (row["start"] + row["end"]) / data1.loc[index, 'seq']

feature = feature.join(mylist)

sequences = data3['seq']
labels = data3['label'].values
print(labels)

pat = re.compile('[AGCTagct]')


def pre_process(text):
    text = pat.findall(text)
    text = [each.lower() for each in text]
    return text


x = sequences.apply(pre_process)

word_set = set()

for lst in x:
    for word in lst:
        word_set.add(word)

word_list = list(word_set)
word_index = dict([(each, word_list.index(each) + 1) for each in word_list])

text = x.apply(lambda x: [word_index.get(word, 0) for word in x])

text_len = 1200

pad_text = [l + (text_len - len(l)) * [0] if len(l) < text_len else l[:text_len] for l in text]
pad_text = np.array(pad_text)
pad_text

pad_text, labels = torch.LongTensor(pad_text), torch.LongTensor(labels)
x_train, x_test, y_train, y_test = train_test_split(pad_text, labels, test_size=0.3)


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, text_list, label_list):
        self.text_list = text_list
        self.label_list = label_list

    def __getitem__(self, index):
        text = torch.LongTensor(self.text_list[index])
        label = self.label_list[index]
        return text, label

    def __len__(self):
        return len(self.text_list)


data_ds = Mydataset(pad_text, labels)

batch_size = 16

data_dl = torch.utils.data.DataLoader(data_ds, batch_size=batch_size, shuffle=False)

# embed_dim = 2 ** (int(np.log2(len(word_list) ** 0.25)) + 2)

embed_dim = 50
hidden_size = 20


class Net(nn.Module):
    def __init__(self, word_list, embed_dim, hidden_size, num_layers=2):
        super().__init__()

        self.em = nn.Embedding(len(word_list) + 1, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers)
        self.pool1 = nn.MaxPool1d(5)
        self.pool2 = nn.MaxPool1d(3)
        self.linear1 = nn.Linear(hidden_size, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, inputs):
        x = self.em(inputs)  # inputs is input, size (seq_len, batch, input_size)               #[16,1200,50]
        x = x.float()
        x, _ = self.gru(x)  # x is outuput, size (seq_len, batch, hidden_size)                  #[16,1200,20]
        x = self.pool1(x)  # [16,1200,4]
        x = self.pool2(x)  # [16,1200,1]
        x = x.view(-1, 1200)
        return x


model = Net(word_list, embed_dim, hidden_size)
model = model.to(device)

loss = nn.CrossEntropyLoss()
loss = loss.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with torch.no_grad():
    outputs = []
    for x, y in data_dl:
        out = model(x)
        outputs.append(out)

outputs1 = torch.concat(outputs, axis=0)

deep_fea = pd.DataFrame(outputs1)

fea_all = feature.join(deep_fea)

X, y, X_test = ori_fea, oy, fea_all
rf_clf = joblib.load('model.pkl')
X_test.columns = X_test.columns.astype(str)

y_pred = rf_clf.predict(X_test)

result_df = pd.DataFrame({
    'Index': X_test.index,
    'Predict_result': np.where(y_pred == 1, 'ncRNA', 'cRNA')
})

print(result_df)
print('finish!')