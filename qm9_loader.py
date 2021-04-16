import deepchem as dc

def load_qm9():

    input_file = 'qm9.csv'
    QM9_TASKS = [
        "homo", "lumo", "gap"
    ]
    featurizers = dc.feat.CoulombMatrix(max_atoms=29)
    loader = dc_data.SDFLoader(QM9_TASKS, featurizer=featurizers)
    dataset = loader.create_dataset(input_file)

    train_dataset, valid_dataset, test_dataset = dc.splits.Splitter.train_valid_test_split(dataset, frac_train = 0.7, frac_valid= 0.1, frac_test = 0.2)

    return train_dataset, valid_dataset, test_dataset