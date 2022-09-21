import glob
import scipy.io as sio
from data3d.utils import *
import logging

def get3Data():
    features = []
    labels = []
    for xfile in glob.glob("../../../data/pos_ideal/features/*input*.mat", recursive=True):
        d = sio.loadmat(xfile)
        d = d["radarImage" if "radarImage" in d.keys() else "idealImage"]
        features.append(d)
        d = sio.loadmat(xfile[:24]+"labels/output"+xfile[38:])
        d = d["radarImage" if "radarImage" in d.keys() else "idealImage"]
        labels.append(d)
    features = np.array(features)
    labels = np.array(labels)
    data = np.concatenate([features, labels], axis=0)
    # Standardization. Rescale data to have a mean of 0, and a standard deviation of 1.
    data = (data - data.mean()) / data.std()
    data = data - data.min()
    features = data[0:(data.shape[0]//2)]
    labels = data[(data.shape[0]//2):]
    logging.info(f"Data: {bcolors.OKGREEN}{features.shape}{bcolors.ENDC}")
    return features, labels

if __name__=="__main__":
    features, labels = get3Data()
    print(features.shape, labels.shape)
    for i in range(features.shape[0]):
        sio.savemat(f"../../../data/data3d/data3d-{i}.mat", {"features": features[i], "labels": labels[i]})
