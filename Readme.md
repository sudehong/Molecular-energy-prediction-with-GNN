#Molecular energy prediction with GNN
(A) Extraction of molecular energies
========
(A1) The energy of each molecule is calculated by density functional theory and the result is located in /A/done/[0-9][0-9]/. There are a total of 12484 molecules in 13 folders from 00-12.Due to the space problem, only a part of the sample can be uploaded. The last one used in step C is another dataset QM9
A_write.py will store the extracted data(The energy of each molecule) into the test.txt file, and A_loder.py will load the data from test.txt. 

(B) Get information about atoms and bonds of molecules.
========
Read the smile format of molecules. Convert with rdkit  B.py

(C) After the last step, the SMILES and ENERGY are converted into a dataset that can be handled by pytorch geometric.
========
Save in the QM9 dataset format ,the dataset format like https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html

>from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data)
>>class QM9(InMemoryDataset):
 >>> data1 = Data(x=data1.x, z=data1.z, pos=data1.pos,edge_index=data1.edge_index,      edge_attr=data1.edge_attr,y=data1.y, name=data1.name, idx=data1.idx)
data_list.append(data1)
torch.save(self.collate(data_list), self.processed_paths[0])

Among them, x is a one-hot vector representing an atom, edge_index is the edge information of the molecular graph, edge_attr is a one-hot vector representing the edge, i.e., the order of the bond, and y is the total energy of the molecule.
Convert smiles to graph of Networkx and convert it to pytorch geometry data. The function of smiles2graph is the number of the executives, and the purpose is that the number of the host number is the one hot anchor, and it is a contact type type type, and the road way, etc.

networkx: https://networkx.org/documentation/stable/
torch-geometric:https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs

(D) GNN model
====
Using the feature vector x_i of the atom from after Conv, calculate the inner product x_i dot x_j between the bound atoms, and regress y (the rest of y) on the sum.


(E)result
======
The loss function does decrease and maintains at a low point and then stops changing. But since multiple molecular data were fitted using one W, the error is large. Please see the picture for the results.


=======
Before starting the training, please install the required packages.
