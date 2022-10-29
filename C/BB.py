import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem import Draw
from rdkit.Chem import rdmolops
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
mol = Chem.MolFromSmiles("CC(CC)C")

df = pd.read_csv('data1.csv')
print(df)