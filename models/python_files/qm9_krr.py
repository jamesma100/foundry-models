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

def Transform_Data():
    
    SMILESdata, GapData = readCSV('../data/qm9.csv')
    SMILESstrings = list(SMILESdata.keys())
    FeatureVector = []
    BadParticles = []
    for string in SMILESstrings:
        mol = Chem.MolFromSmiles(SMILESdata[string])
        if(mol != None):
            fp = MACCSkeys.GenMACCSKeys(mol)
            fpBits = fp.ToBitString()
            FeatureVector.append(split(fpBits))
        else:
            BadParticles.append(string)
    X = np.array(FeatureVector[:5000])
    Z = np.array(FeatureVector[5000:7500])
    print(X.shape)
    print(Z.shape)
    for keys in BadParticles:
        del GapData[keys]
    GapData = list(GapData.values())
    y3 = np.array(GapData[:5000])
    w3 = np.array(GapData[5000:7500])
    return X, Z, w3, y3

def split(bitString):
    
    return [int(char) for char in bitString] 

def readCSV(filepath):

    Smile_Strings = {}
    Gap_Id = {}
    with open(filepath, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Smile_Strings[row['mol_id']] = row['smiles']
            Gap_Id[row['mol_id']] = float(row['gap'])
    
    return Smile_Strings, Gap_Id
    

    
def Create_Model():
    
    model3 = KernelRidge(kernel='laplacian')
    X, Z, w3, y3= Transform_Data()
    model3.fit(X, y3)
    predict3 = model3.predict(Z)
    '''
    c = np.random.rand(25000)
    pyplot.scatter(w3, predict3, c=c)
    pyplot.ylabel('Test Data')
    pyplot.xlabel('Predict Data')
    pyplot.title('Gap')
    pyplot.show()
    '''
    cv_results = cross_validate(model3, X, y3, cv=3)
    sorted(cv_results.keys())
    print(cv_results['test_score'])
    
    R_squared3 = r2_score(w3, predict3)
    RMSE3 = math.sqrt(mean_squared_error(w3, predict3))
    MAE3 = mean_absolute_error(w3, predict3)
    print("The R^2 score of GAP model is: ",R_squared3)
    print("The RMSE score of GAP model is: ", RMSE3)
    print("The MAE score of GAP model is: ", MAE3)
            
if __name__ == "__main__":
    Create_Model()

