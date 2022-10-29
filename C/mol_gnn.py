import numpy as np
import  pandas as pd
import  networkx as nx
import rdkit
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import torch
import  torch_geometric
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.nn import GCNConv
import tqdm
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, global_add_pool
import matplotlib.pyplot as plt

data = pd.read_csv("\\raw\\qm9 -1.csv ")
# print(data)
smiles =  data['smiles']
smi = np.array(smiles)

energy1 =  data['h298_atom']
energy = np.array(energy1)

def smiles2graph(smile,kekulize=False):
    '''
          Parameters
          ----------
          smiles: String

          Returns
          -------
          g: nx.Graph
          action_index: Optional[int]

          '''
    boolean = {True: 1, False: 0}
    bondtype = {'SINGLE': 1, 'DOUBLE': 2, 'AROMATIC': 1.5, 'TRIPLE': 3}

    mol = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol)
    # if mol is None:
    #     raise ValueError('failed to parse smiles: %s' % smile)
    # elif kekulize:
    #     Chem.Kekulize(mol)
    # else:
    #     pass

    am2nid = {}
    #create nodes
    nodes = []
    sy = []
    for iatom,atom in enumerate(mol.GetAtoms()):
        idx = atom.GetIdx() #获取原子id
        am2nid[atom.GetAtomMapNum()] = idx ##  unmapped atom returns 0
        symbol = atom.GetSymbol() #获取原子的元素符号
        if atom.GetIsAromatic():
            Hybridization = 'Aromatic'
        elif symbol == 'H':
            Hybridization = 's'
        else:
            Hybridization = str(atom.GetHybridization())
        nodes.append((idx, {'symbol': symbol, 'hybridization': Hybridization}))
        sy.append(symbol)

    #create edges
    edges = []
    b = []
    for bond in mol.GetBonds():
         source = bond.GetBeginAtomIdx()
         target = bond.GetEndAtomIdx()
         order = bond.GetBondTypeAsDouble()
         edges.append((source, target,{'order': str(order)}))
         b.append(bond.GetBondType())

    # print(nodes)
    #node and edges is list
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    #节点类型（不重复）
    # s = list(set(sy))
    s = ['C','H','O','N','F']
    #边类型（不重复）
    bb = [rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,rdkit.Chem.rdchem.BondType.TRIPLE ,rdkit.Chem.rdchem.BondType.AROMATIC]
    # print(s)
    t = list(g.edges)
    #添加x
    for k1 in range(len(g.nodes)):
        ss = g.nodes[k1]['symbol']
        #set中元素下标
        i = s.index(ss)
        g.nodes[k1]['x']  = [1 if k ==i else 0 for k in range(len(s))]
    #添加edges_atrr：边的类型的onehot

    edge_attr = []
    for i in range(len(g.edges)):
        #每次取边的类型
        e = b[i]
        n1 = t[i][0]
        n2 = t[i][1]
        #set中元素下标
        j = bb.index(e)
        l = [1 if k ==j else 0 for k in range(len(bb))]
        edge_attr.append(l)
        edge_attr.append(l)

    return (g,edge_attr,mol)


class QM9(InMemoryDataset):

    def __init__(self,root,transform=None, pre_transform=None):
        super(QM9, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        data_list = []
        for i in range(len(smi)):
            #访问smi列表里的元素
            # try:
            #     q = smiles2graph(smi[i])
            # except Exception as e:
            #     print(smi[i])
            # continue
            q = smiles2graph(smi[i])
            G = q[0]
            y = energy[i]
            data1 = from_networkx(G)
            edge_attr = q[1]
            mol = q[2]

            # data3 = Data(x=data1.x,
            #              edge_index=data1.edge_index, edge_attr=edge_attr,
            #              y=y)
            data3 = Data(x=torch.tensor(data1.x,dtype=torch.float32),
                         edge_index=torch.tensor(data1.edge_index), edge_attr=edge_attr,
                         y=torch.tensor(y,dtype=torch.float32))
            # print(data3.x)
            data_list.append(data3)

        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
        torch.save(self.collate(data_list), self.processed_paths[0])



def calcMeanStd(dataset):
  from sklearn.linear_model import LinearRegression
  ndata =  len(dataset)
  X = np.zeros((ndata,5),dtype = np.float32)
  y = np.zeros(ndata,dtype = np.float32)
  for idata,data1 in enumerate(dataset[0:]):
    X[idata,:] = np.sum(data1.x[:,0:5].numpy(),axis=0)
    y[idata] = data1.y
  reg = LinearRegression(fit_intercept=False).fit(X, y)
  dy = y - reg.predict(X)
  # print('linear regression',reg.score(X, y), reg.coef_, reg.intercept_, dy.mean(),dy.std())
  return reg.coef_[:],dy.std()

dataset = QM9(root="C:\\Users\\SU DEHONG\\Desktop\\test\\finalData\\data")
n = (len(dataset) + 9) // 10
# Normalize targets to mean~0, std~1.
coef,std = calcMeanStd(dataset[2 * n:])
# print('coef,std=',coef,std)

class MyTransform(object):
  def __call__(self, data):
    xsum = data.x[:,0:5].sum(axis=0)
    data.y = (data.y - sum(xsum * coef)) / std
    return data

transform = T.Compose([MyTransform()])

# a,b = calcMeanStd(dataset)
# print(a,b)

# dataset = QM9(root="C:\\Users\\SU DEHONG\\Desktop\\test\\finalData\\data",transform=transform)
dataset = QM9(root="C:\\Users\\SU DEHONG\\Desktop\\test\\finalData\\data")

print(dataset.data)
print(np.shape(dataset.data))
print(dataset.data.y)
print(dataset[0])
print(dataset[0].edge_index)
# print(dataset.num_features)
# print(dataset.data.edge_attr)
# print(dataset.num_node_features)


class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(5,64)
        self.conv2 = GCNConv(64, 5)
        # self.lin = nn.Linear(20000,20000,bias=0)
        self.w = torch.nn.Parameter(torch.rand(1, dtype=torch.float32))

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x,edge_index)
        x = self.conv2(x, edge_index)
        # print(x.shape,data[0])

        sum = []
        for idata, data1 in enumerate(data[0:]):
            ss = []
            for i in range(len(data1.edge_index[0])):
                source = data1.edge_index[0][i]
                target = data1.edge_index[1][i]

                left = data1.x[source, 0:].detach().numpy()
                right = data1.x[target, 0:].detach().numpy()
                s = np.dot(left, right)
                ss.append(s)
            sum.append(np.sum(ss))
        x = torch.tensor(sum, dtype=torch.float32)
        # print(x.shape, x);raise RuntimeError
        # dataset1 = dataset.copy()
        # dataset1.data.x = x
        # sum = []
        # for idata, data1 in enumerate(dataset1[0:]):
        #     # 原子間で内積 x_i dot x_j を計算し
        #     ss = []
        #     for i in range(len(data1.edge_index[0])):
        #         source = data1.edge_index[0][i]
        #         target = data1.edge_index[1][i]
        #
        #         left =  data1.x[source,0:].detach().numpy()
        #         right = data1.x[target,0:].detach().numpy()
        #         s = np.dot(left,right)
        #         ss.append(s)
        #     sum.append(np.sum(ss))
        # x = torch.tensor(sum,dtype=torch.float32)
        out = x*self.w
        return out

# model = GNN()
# result = model(dataset.data)
# print(result)

def train():
  model.train()
  loss_all = 0
  for data in train_loader:
    data2 = data.to(device)
    optimizer.zero_grad()
    y_pred = model(data2).to(device)
    loss = criterion(y_pred, data2.y)
    loss.backward()
    loss_all += loss.item() * data2.y.size(0)
    optimizer.step()
  return loss_all / len(train_dataset)


def test(loader):
  model.eval()
  error = 0
  for data in loader:
    data3 = data.to(device)
    error += (model(data3) * std - data3.y * std).abs().sum().item()  # MAE
  return error / len(loader.dataset)


# Split datasets.
dataset = dataset.shuffle()
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]


test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

dim = 64
model = GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5,
                                                       min_lr=0.00001)
best_val_error = None
loss_hist = []
val_hist = []
test_hist = []
for epoch in range(1, 100):
    lr = scheduler.optimizer.param_groups[0]['lr']
    train_loss = train()
    train_error = test(train_loader)
    val_error = test(val_loader)
    scheduler.step(val_error)
    loss_hist.append(train_loss)
    val_hist.append(val_error)
    test_hist.append(val_error)

    if best_val_error is None or val_error <= best_val_error:
        test_error = test(test_loader)
        best_val_error = val_error

    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Train MAE: {:.7f}, Validation MAE: {:.7f}, '
          'Test MAE: {:.7f}'.format(epoch, lr, train_loss, train_error, val_error, test_error), flush=True)

model.eval()


import matplotlib.pyplot as plt

plt.plot(loss_hist, label="Train Loss")
plt.plot(val_hist, label="Val Loss")
plt.plot(test_hist, label="Test Loss")
plt.yscale('log')
plt.legend()
plt.show()



