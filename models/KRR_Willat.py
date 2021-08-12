
import torch

from dscribe.descriptors import SOAP
from ase.io import read
from ase import Atoms
from ase.build import molecule
from sklearn.kernel_ridge import KernelRidge
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import sqlite3
import numpy as np
import csv
import math
from sklearn.model_selection import cross_validate
import random

def parse_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        base, power = s.split('*^')
        return float(base) * 10**float(power)

'''
xyz format
- line 1: number of atoms n
- line 2: scalar properties
- line 3, ..., n+1: element type, coordinates xyz, Mulliken partial charges on atoms
'''

# parse an xyz file, returns a result dictionary of a molecule
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
                scalar_properties = [parse_float(i) for i in line.split()[2:]]
            elif 2 <= line_num <= 1 + num_atoms:
                atom_symbol, x, y, z, charge = line.split()
                atomic_symbols.append(atom_symbol)
                xyz.append([parse_float(x), parse_float(y), parse_float(z)])
                charges.append(parse_float(charge))
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

def build_soap(start, end, u0_data):
    molecules_unshuffled = []
    for i in range(start, end+1):
        # create molecule object
        file_name = "XYZ_files/dsgdb9nsd_" + \
            str(i).zfill(6) + ".xyz"
        molecule = parse_xyz(file_name)
        molecule_obj = Atoms(symbols=molecule["atomic_symbols"], positions=molecule["pos"])
        molecules_unshuffled.append(molecule_obj)
    temp = list(zip(molecules_unshuffled, u0_data[:end]))
    random.shuffle(temp)
    molecules, u0_Data_shuffled = zip(*temp)
        
    # create SOAP object
    soap = SOAP(
        species=["C", "H", "O", "N", "F"],
        periodic=False,
        rcut=5,
        nmax=12,
        lmax=9,
        average="outer",
        sparse=False
    )
    
    feature_vector = soap.create(molecules[:625])
    return feature_vector, u0_Data_shuffled[:625]

def transform(filepath):
    # parse gap data from qm9
    u0 = {}
    with open(filepath, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            u0[row['mol_id']] = float(row['u0_atom'])
    u0_data_unshuffle = list(u0.values())
    soaps, u0_data= build_soap(1, 133000, u0_data_unshuffle)
    
    
    
    u0_train, u0_test = np.array(u0_data[:500]), np.array(u0_data[500:625])
    soap_train, soap_test = np.array(soaps[:500],dtype=object), np.array(soaps[500:],dtype=object)
    sd_train = np.std(u0_train)
    print("SD of training data: ",sd_train)
    return soap_train, soap_test, u0_train, u0_test

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
    soap_train, soap_test, u0_train, u0_test = transform("data/qm9.csv")
    model1.fit(soap_train, u0_train)
    predict1 = model1.predict(soap_test)
    predict = list(predict1)
    test_data = list(u0_test)
    #model2.fit(X, y2)
    #predict2 = model2.predict(Z)
    #model3.fit(X, y3)
    #predict3 = model3.predict(Z)
    
    #c = np.random.rand(250)
    #pyplot.scatter(test_data, predict, c=c)
    #pyplot.ylabel('Test Data')
    #pyplot.xlabel('Predict Data')
    #pyplot.title('Atomization Energy')
    #pyplot.show()
    
    R_squared = r2_score(u0_test, predict1)
    RMSE = math.sqrt(mean_squared_error(u0_test, predict1))
    MAE = mean_absolute_error(u0_test, predict1)
    sd_actual = np.std(u0_test)
    sd_predict = np.std(predict1)
    print("SD of actual data: ",sd_actual)
    print("SD of predicted: ",sd_predict)
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


