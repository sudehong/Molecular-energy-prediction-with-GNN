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

data = pd.read_csv("C:\\Users\\SU DEHONG\\Desktop\\test\\finalData\\data\\raw\\qm9.csv")
# print(data)
smiles =  data['smiles']
smi = np.array(smiles)

energy1 =  data['h298_atom']
energy = np.array(energy1)

from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

class MoleculesDataset(InMemoryDataset):
    def __init__(self, smiles, ys, transform = None):
        super().__init__('.', transform)

        boolean = {True:1, False:0}
        hybridization = {'SP':1, 'SP2':2, 'SP3':3, 'SP3D':3.5}
        bondtype = {'SINGLE':1, 'DOUBLE':2, 'AROMATIC':1.5, 'TRIPLE':3}

        datas = []
        for smile, y in zip(smiles, ys):
            mol = Chem.MolFromSmiles(smile)

            embeddings = []
            for atom in mol.GetAtoms():
                a = []
                a.append(atom.GetAtomicNum())
                a.append(atom.GetMass())
                a.append(hybridization[str(atom.GetHybridization())])
                a.append(boolean[atom.IsInRing()])
                a.append(boolean[atom.GetIsAromatic()])
                embeddings.append(a)
            embeddings = torch.tensor(embeddings)

            edges = []
            edge_attr = []
            for bond in mol.GetBonds():
                edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                b = []
                b.append(bondtype[str(bond.GetBondType())])
                b.append(boolean[bond.GetIsAromatic()])
                b.append(boolean[bond.IsInRing()])
                edge_attr.append(b)
            edges = torch.tensor(edges).T
            edge_attr = torch.tensor(edge_attr)

            y = torch.tensor(y, dtype=torch.long)

            data = Data(x=embeddings, edge_index=edges, y=y, edge_attr=edge_attr)
            datas.append(data)

        self.data, self.slices = self.collate(datas)

max_nodes = 128
dataset = MoleculesDataset(smi, energy, transform=torch_geometric.transforms.ToDense(max_nodes))
dataset.data

print(dataset.data)
print(dataset.data.x)