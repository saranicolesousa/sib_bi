import pandas as pd
from si.data.dataset import Dataset

def read_csv(filename: str, sep: str = ',', features: bool = False, label: bool = False):
    df=pd.read_csv(filename, sep=sep)

    if features:
        if label:
            X = df.iloc[:, :-1].to_numpy()
            y = df.iloc[:, -1].to_numpy()
            feature_names = list(df.columns[:-1])
            label_name = df.columns[-1]
        else:
            X = df.to_numpy()
            y = None
            feature_names = list(df.columns)
            label_name=None
    elif not features:
        if label:
            X = df.iloc[:, :-1].to_numpy()
            y= df.iloc[:, -1].to_numpy()
            feature_names = None
            label_name = None
        else:
            X = df.to_numpy()
            y = None
            feature_names = None
            label_name = None

    return Dataset(X=X, y=y, features=feature_names, label=label_name)

def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = False, label: bool = False):
    if features:
        column_names = dataset.features
    else:
        column_names = None

    df = pd.DataFrame(dataset.X, columns=column_names)

    if label:
        if not dataset.has_label():
            raise ValueError("Dataset does not have a label")
        if features:
            label_name = dataset.label
        else:
            label_name = 'y'
        df[label_name] = dataset.y

    df.to_csv(filename, sep=sep, index=False)



