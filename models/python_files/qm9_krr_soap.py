import numpy as np

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
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.model_selection import cross_validate

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

# given 1 <= start <= 133885 and 1 <= end <= 133885 return array of soap 
# vectors for molecules start, start+1, ..., end
def build_soap(start, end):
    soaps = []
    for i in range(start, end+1):

        # create molecule object
        file_name = "../../data/dsgdb9nsd.xyz/dsgdb9nsd_" + \
            str(i).zfill(6) + ".xyz"
        molecule = parse_xyz(file_name)
        molecule_obj = Atoms(symbols=molecule["atomic_symbols"], positions=molecule["pos"])

        # set up soap descriptor
        species = set()
        species.update(molecule_obj.get_chemical_symbols())

        soap = SOAP(
            species=species,
            periodic=False,
            rcut=5,
            nmax=8,
            lmax=8,
            average="outer",
            sparse=False
        )
        feature_vector = soap.create(molecule_obj)
        soaps.append(feature_vector)
    return soaps
    
def build_soap2(start, end):
    molecules = []
    for i in range(start, end+1):
        # create molecule object
        file_name = "../../data/dsgdb9nsd.xyz/dsgdb9nsd_" + \
            str(i).zfill(6) + ".xyz"
        molecule = parse_xyz(file_name)
        molecule_obj = Atoms(symbols=molecule["atomic_symbols"], positions=molecule["pos"])
        molecules.append(molecule_obj)
        
    # create SOAP object
    soap = SOAP(
        species=["C", "H", "O", "N", "F"],
        periodic=False,
        rcut=5,
        nmax=9,
        lmax=9,
        average="outer",
        sparse=False
    )
    
    feature_vector = soap.create(molecules)
    return feature_vector

def transform(filepath):
    # parse gap data from qm9
    gap = {}
    with open(filepath, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            gap[row['mol_id']] = float(row['gap'])
    gap_data = list(gap.values())[:10000]
    soaps = build_soap2(1, 10000)
    
    gap_train, gap_test = np.array(gap_data[:8000]), np.array(gap_data[8000:])
    soap_train, soap_test = np.array(soaps[:8000],dtype=object), np.array(soaps[8000:],dtype=object)
    return soap_train, soap_test, gap_train, gap_test

def create_model():
    model = KernelRidge(kernel='laplacian')
    soap_train, soap_test, gap_train, gap_test = transform("../../data/qm9.csv")
    model.fit(soap_train, gap_train)
    predict = model.predict(soap_test)
    
    r_sqr = r2_score(gap_test, predict)
    rmse = math.sqrt(mean_squared_error(gap_test, predict))
    mae = mean_absolute_error(gap_test, predict)
    #print("gap test: ", gap_test)
    #print("gap predict: ", predict)
    print("r^2: ",r_sqr)
    print("rmse: ", rmse)
    print("mae: ", mae)

if __name__ == "__main__":
    create_model()
