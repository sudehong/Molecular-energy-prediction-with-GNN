from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
m = Chem.MolFromSmiles('CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3')
m3d=Chem.AddHs(m)
AllChem.EmbedMolecule(m3d, randomSeed=1)
Draw.MolToImage(m3d, size=(250,250))