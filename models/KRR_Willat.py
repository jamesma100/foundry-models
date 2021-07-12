from sklearn.kernel_ridge import KernelRidge
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import sqlite3
import numpy as np
import csv
import math
from ase.io.xyz import read_xyz
from ase.io import read
from ase.build import molecule
from ase import Atoms
from sklearn.model_selection import cross_validate
from os import listdir
from os.path import isfile, join
from dscribe.descriptors import SOAP
import torch

'''
xyz format
- line 1: number of atoms n
- line 2: scalar properties
- line 3, ..., n+1: element type, coordinates xyz, Mulliken partial charges on atoms
'''
def parse_xyz(filename):
    num_atoms = 0
    scalar_properties = []
    atomic_symbols = []
    xyz = []
    charges = []
    smiles = ""
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num == 0:
                num_atoms = int(line)
            elif line_num == 1:
                scalar_properties = [float(i) for i in line.split()[2:]]
            elif 2 <= line_num <= 1 + num_atoms:
                atom_symbol, x, y, z, charge = line.split()
                atomic_symbols.append(atom_symbol)
                xyz.append([float(x), float(y), float(z)])
                charges.append(float(charge))
            elif line_num == num_atoms + 3:
                smiles = str(line.split())

    result = {
        'num_atoms': num_atoms,
        'atomic_symbols': atomic_symbols,
        'pos': torch.tensor(xyz),
        'charges': np.array(charges),
        'smiles': smiles
    }
    return result

def Transform_Data():
    
    molecules = []
    num_files = 133885
    temp = 10
    files = listdir("XYZ_files")
    print(files)
    for file_name in files:
    
        # create molecule object
        molecule = parse_xyz("XYZ_files/"+file_name)
        molecule_obj = Atoms(symbols=molecule["atomic_symbols"], positions=molecule["pos"])
        molecules.append(molecule_obj)

    # set up soap descriptor
    species = set()
    species.update(molecule_obj.get_chemical_symbols())

    soap = SOAP(
        species=["C", "H", "O", "N", "F"],
        periodic=False,
        rcut=5,
        nmax=8,
        lmax=8,
        average="outer",
        sparse=False
    )
    print(molecules)
    soap_train = soap.create(molecules[:25000])
    soap_test = soap.create(molecules[25000:50000])
    print(soap_train.shape)
    
    
    SMILESdata, HomoData, GapData, Atomization_E = readCSV('data/qm9.csv')
    Atomization_E = list(Atomization_E.values())
    y = np.array(Atomization_E[:25000])
    w = np.array(Atomization_E[25000:50000])
    
    return soap_train, soap_test, y, w

def split(bitString):
    
    return [int(char) for char in bitString] 

def readCSV(filepath):

    Smile_Strings = {}
    Homo_Id = {}
    Gap_Id = {}
    Atomization_Id = {}
    with open(filepath, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Smile_Strings[row['mol_id']] = row['smiles']
            Gap_Id[row['mol_id']] = float(row['gap'])
            Homo_Id[row['mol_id']] = float(row['homo'])
            Atomization_Id[row['mol_id']] = float(row['u0'])
    
    return Smile_Strings, Homo_Id, Gap_Id, Atomization_Id
    

    
def Create_Model():
    
    model1 = KernelRidge(kernel='laplacian')
    #model2 = KernelRidge(kernel='laplacian')
    #model3 = KernelRidge(kernel='laplacian')
    soap_train, soap_test, y ,w = Transform_Data()
    model1.fit(soap_train, y)
    predict1 = model1.predict(soap_test)
    #model2.fit(X, y2)
    #predict2 = model2.predict(Z)
    #model3.fit(X, y3)
    #predict3 = model3.predict(Z)
    
    c = np.random.rand(25000)
    pyplot.scatter(w, predict1, c=c)
    pyplot.ylabel('Test Data')
    pyplot.xlabel('Predict Data')
    pyplot.title('Atomization Energy')
    pyplot.show()
    
    R_squared = r2_score(w, predict1)
    RMSE = math.sqrt(mean_squared_error(w, predict1))
    MAE = mean_absolute_error(w, predict1)
    print("The R^2 score of Atomization_E model is: ",R_squared)
    print("The RMSE score of Atomization_E model is: ", RMSE)
    print("The MAE score of Atomization_E model is: ", MAE)
'''
    c = np.random.rand(25000)
    pyplot.scatter(w1, predict1, c=c)
    pyplot.ylabel('Test Data')
    pyplot.xlabel('Predict Data')
    pyplot.title('Homo')
    pyplot.show()
    
    R_squared1 = r2_score(w1, predict1)
    RMSE1 = math.sqrt(mean_squared_error(w1, predict1))
    MAE1 = mean_absolute_error(w1, predict1)
    print("The R^2 score of HOMO model is: ",R_squared1)
    print("The RMSE score of HOMO model is: ", RMSE1)
    print("The MAE score of HOMO model is: ", MAE1)

    c = np.random.rand(25000)
    pyplot.scatter(w3, predict3, c=c)
    pyplot.ylabel('Test Data')
    pyplot.xlabel('Predict Data')
    pyplot.title('Gap')
    pyplot.show()
    
    R_squared3 = r2_score(w3, predict3)
    RMSE3 = math.sqrt(mean_squared_error(w3, predict3))
    MAE3 = mean_absolute_error(w3, predict3)
    print("The R^2 score of GAP model is: ",R_squared3)
    print("The RMSE score of GAP model is: ", RMSE3)
    print("The MAE score of GAP model is: ", MAE3)
'''

if __name__ == "__main__":
    Create_Model()


