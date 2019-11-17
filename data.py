from torch.utils.data import Dataset
import pickle
import pandas as pd
import logging
from sklearn.model_selection import KFold, train_test_split

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Experiment:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        split = TrainTestSplit(X, y)
        self.train_dataset = BasicDataset(split.X_train, split.y_train)
        self.test_dataset = BasicDataset(split.X_test, split.y_test)

class BasicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        sample = self.X[index]
        sample_label = self.y[index]
        return sample, sample_label

class TrainTestSplit:
    def __init__(self, X, y, test_size=0.2):
        super(TrainTestSplit, self).__init__()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


class PriorData(Dataset):
    def __init__(self, prior_data):
        self.data = prior_data
        pass

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return self.features.shape[0]

    def ensure_overlap_with_response(self, response_data):
        raise NotImplementedError

    @staticmethod
    def load_data():
        raise NotImplementedError


class MergeData(PriorData):
    def __init__(self, merge_data):
        super(MergeData, self).__init__(merge_data)

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_feature(self, feature):
        return self.data.loc[feature]

    @staticmethod
    def load_data():
        # MERGE dataframe comes with driver features as columns, and genes as rows.  We want the opposite to conform
        # to the general patients as rows, genes as columns scheme
        logging.info("Loading merge data")
        df = pd.read_csv("merge_features.csv")
        df = df.transpose()
        new_header = df.iloc[0]  # grab the first row for the header
        df = df[1:]  # take the data less the header row
        df.columns = new_header  # set the header row as the df header
        df = df.astype(float)
        logging.info("Merge data loaded")
        return MergeData(df)


class ResponseData(Dataset):
    def __init__(self, X, y):
        super(ResponseData, self).__init__()
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        sample = self.X.iloc[idx].values
        sample_label = self.y.iloc[idx].values
        return sample, sample_label

    def __len__(self):
        return self.X.shape[0]

    def ensure_overlap_with_prior(self, prior_features):
        raise NotImplementedError

    @staticmethod
    def load_data():
        raise NotImplementedError


class ExVivoDrugData(ResponseData):

    def __init__(self, X, y):
        super(ExVivoDrugData, self).__init__(X=X, y=y)
        self.drug_columns = [col for col in self.X.columns if 'drug_' in col]

    # Split by patient
    def kfold_patient_split(self, n_splits):
        logging.info("Splitting drug response data into {} folds".format(n_splits))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        patient_ids = self.X.patient_id.unique()
        train_test_splits = []

        for train_patient_idxs, test_patient_idxs, in kf.split(patient_ids):

            train_patient_ids = patient_ids[train_patient_idxs]
            test_patient_ids = patient_ids[test_patient_idxs]

            X_train, y_train = self.data_for_patients(train_patient_ids)
            X_test, y_test = self.data_for_patients(test_patient_ids)

            train_test_splits.append(
                TrainTestSplit(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                )
            )
        logging.info("Splitting complete")
        return train_test_splits

    def data_for_patients(self, patient_ids):
        X_for_patients = self.X[self.X.patient_id.isin(patient_ids)].drop(columns="patient_id")
        y_for_patients = self.y[self.y.patient_id.isin(patient_ids)].drop(columns="patient_id")
        return X_for_patients, y_for_patients

    #FIXME
    # We're loading the patient ID here, but then tossing it once we look at specific train/test splits.
    # Would be nice to have a better abstraction that captures this, so that we don't accidentally have ids
    # in places we shouldn't (or vice-versa)
    @staticmethod
    def load_data():
        logging.info("Loading drug response data")
        DATADIR = '../fdata/ohsu_data/'
        final_frame = pickle.load(open(DATADIR + 'final_frame.p', 'rb'))
        X = pickle.load(open(DATADIR + 'X_rna_seq_final.p', 'rb'))
        y = final_frame[["patient_id", "IC50"]]
        logging.info("Drug response data loaded")
        return ExVivoDrugData(X=X, y=y)

