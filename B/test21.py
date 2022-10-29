from rdkit import rdBase, Chem
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole
import sys, py3Dmol
import pandas as pd

smi='CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3'
mol = Chem.MolFromSmiles(smi)
main_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
atoms_info = [ (atom.GetIdx(), atom.GetAtomicNum(), atom.GetSymbol()) for atom in mol.GetAtoms()]
atom_index= list(range(len(atoms_info)))

bonds_info=[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
g = nx.Graph()
g.add_nodes_from(atom_index)
g.add_edges_from(bonds_info)
plt.figure(figsize=(10,8))
nx.draw_networkx(g,node_size=1000, width=3)
plt.show()
