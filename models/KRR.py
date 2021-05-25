from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import sqlite3
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys

def Transform_Data():
    
    SMILESdata, Atomizationdata = readSqliteTable()
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
    X = np.array(FeatureVector[:25000])
    Z = np.array(FeatureVector[25000:50000])
    print(X.shape)
    print(Z.shape)
    for keys in BadParticles:
        del Atomizationdata[keys]
    AtomizationE = list(Atomizationdata.values())
    y = np.array(AtomizationE[:25000])
    w = np.array(AtomizationE[25000:50000])
    print(y.shape)
    print(w.shape)
    return X, y, Z, w



def split(bitString):
    
    return [int(char) for char in bitString] 
    
def Create_Model():
    
    model = KernelRidge(kernel='laplacian')
    X, y, Z, w = Transform_Data()
    model.fit(X, y)
    predict = model.predict(Z)
    print(predict[1])
    R_squared = r2_score(predict, w)
    RMSE = mean_squared_error(predict, w)
    MAE = mean_absolute_error(predict, w)
    print("The R^2 score is: ",R_squared)
    print("The RMSE score is: ", RMSE)
    print("The MAE score is: ", MAE)
    
    

def readSqliteTable():
    try:
        sqliteConnection = sqlite3.connect('./data/g4mp2-gdb9.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        sqlite_select_query = """SELECT * from text_key_values"""
        cursor.execute(sqlite_select_query)
        records = cursor.fetchall()
        SMILESstrings = {}
        SMILESkeys = {}
        for row in records:
            if(row[0] == 'Smiles'):
                SMILESstrings[row[2]] = row[1]
                
        sqlite_select_query = """SELECT * from number_key_values"""
        cursor.execute(sqlite_select_query)
        records = cursor.fetchall()
        AtomizationE = {}
        AtomizationE2 = {}
        for row in records:
            if(row[0] == 'g4mp2_AtomizationE'):
                AtomizationE[row[2]] = row[1]
        
        
        cursor.close()
        
        
        return SMILESstrings, AtomizationE

    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if (sqliteConnection):
            sqliteConnection.close()
            print("The SQLite connection is closed")
            
Create_Model()
