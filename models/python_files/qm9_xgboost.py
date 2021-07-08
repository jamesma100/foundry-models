import math
import xgboost as xgb
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import sqlite3
import numpy as np
import csv
from rdkit import Chem
from rdkit.Chem import MACCSkeys

def Transform_Data():
    
    SMILESdata, HomoData, LumoData, GapData = readCSV('../data/qm9.csv')
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
    X = np.array(FeatureVector[:75000])
    Z = np.array(FeatureVector[75000:100000])
    print(X.shape)
    print(Z.shape)
    for keys in BadParticles:
        del HomoData[keys]
        del LumoData[keys]
        del GapData[keys]
    HomoData = list(HomoData.values())
    LumoData = list(LumoData.values())
    GapData = list(GapData.values())
    y1 = np.array(HomoData[:75000])
    y2 = np.array(LumoData[:75000])
    y3 = np.array(GapData[:75000])
    w1 = np.array(HomoData[75000:100000])
    w2 = np.array(LumoData[75000:100000])
    w3 = np.array(GapData[75000:100000])
    print(y1.shape)
    print(w1.shape)
    return X, Z, w1, w2, w3, y1, y2, y3

def split(bitString):
    
    return [int(char) for char in bitString] 

def readCSV(filepath):

    Smile_Strings = {}
    Homo_Id = {}
    Lumo_Id = {}
    Gap_Id = {}
    with open(filepath, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Smile_Strings[row['mol_id']] = row['smiles']
            Homo_Id[row['mol_id']] = float(row['homo'])
            Lumo_Id[row['mol_id']] = float(row['lumo'])
            Gap_Id[row['mol_id']] = float(row['gap'])
    
    return Smile_Strings, Homo_Id, Lumo_Id, Gap_Id
    

    
def Create_Model():
    
    model1 = xgb.XGBRegressor()
    model2 = xgb.XGBRegressor()
    model3 = xgb.XGBRegressor()
    X, Z, w1, w2, w3, y1, y2, y3= Transform_Data()
    model1.fit(X, y1)
    model2.fit(X, y2)
    model3.fit(X, y3)
    predict1 = model1.predict(Z)
    predict2 = model2.predict(Z)
    predict3 = model3.predict(Z)
    c = np.random.rand(25000)
    pyplot.scatter(w1, predict1, c=c)
    pyplot.ylabel('Test Data')
    pyplot.xlabel('Predict Data')
    pyplot.title('Homo')
    pyplot.xlim(-0.5, 0)
    pyplot.ylim(-0.4, -0.1)
    pyplot.show()
    
    R_squared1 = r2_score(w1, predict1)
    RMSE1 = math.sqrt(mean_squared_error(w1, predict1))
    MAE1 = mean_absolute_error(w1, predict1)
    print("The R^2 score of HOMO model is: ",R_squared1)
    print("The RMSE score of HOMO model is: ", RMSE1)
    print("The MAE score of HOMO model is: ", MAE1)

    c = np.random.rand(25000)
    pyplot.scatter(w2, predict2, c=c)
    pyplot.ylabel('Test Data')
    pyplot.xlabel('Predict Data')
    pyplot.title('Lumo')
    pyplot.xlim(-0.15, 0.15)
    pyplot.ylim(-0.15, 0.3)
    pyplot.show()
    R_squared2 = r2_score(w2, predict2)
    RMSE2 = math.sqrt(mean_squared_error(w2, predict2))
    MAE2 = mean_absolute_error(w2, predict2)
    print("The R^2 score of LUMO model is: ",R_squared2)
    print("The RMSE score of LUMO model is: ", RMSE2)
    print("The MAE score of LUMO model is: ", MAE2)

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
            
if __name__ == "__main__":
    Create_Model()
