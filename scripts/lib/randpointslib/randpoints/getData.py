from absl import flags, app
from absl.flags import FLAGS
import numpy as np
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot
#import h5py
import random
from torch.nn.functional import normalize
import torch
import pymatreader
import mat73

flags.DEFINE_string('datasetPath', 
                    '/work2/07945/chrivas/maverick2/tiftrc/denoising/data/', 
                    'set the path to the dataset.')

flags.DEFINE_string('dataset', 
                    'dataset1', 
                    'set your dataset')

flags.DEFINE_boolean('printDataInfo', 
                    False, 
                    'set your dataset')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# USED Only when h5py is specified!
def printDataInfo(mat_contents, tag, i):
    print(f" > {i} -> mat_contents[{bcolors.OKGREEN+tag+bcolors.ENDC}] : {type(mat_contents[tag])}")
    if isinstance(mat_contents[tag], h5py.Group):
        print(f"{mat_contents[tag].keys()}")
        for k, subtag in enumerate(mat_contents[tag].keys()):
            print(f"   > {i}, {k} -> mat_contents[{bcolors.OKGREEN+tag+bcolors.ENDC}][{bcolors.OKGREEN+subtag+bcolors.ENDC}] : ")
            if isinstance(mat_contents[tag][subtag], h5py.Group):
                for l, t in enumerate(mat_contents[tag][subtag].keys()):
                    if mat_contents[tag][subtag][t] is not None:
                        print(f"     > {i}, {k}, {l} -> mat_contents[{bcolors.OKGREEN+tag+bcolors.ENDC}][{bcolors.OKGREEN+subtag+bcolors.ENDC}][{bcolors.OKGREEN+t+bcolors.ENDC}] : {mat_contents[tag][subtag][t].shape} : {mat_contents[tag][subtag][t].dtype}")
    else:
        print(f" > {i} -> mat_contents[{bcolors.OKGREEN+tag+bcolors.ENDC}] : {mat_contents[tag].shape}")
        
def assignDataset():
    if "dataset1" in FLAGS.dataset:
        d = "sar1DMIMOAWR1243_256_rand.mat"
    elif "dataset2" in FLAGS.dataset:
        d = "sar1DMIMOAWR1243_128_rand.mat"
    elif "dataset3" in FLAGS.dataset:
        d = "sar1DMIMOAWR1243_128_rand_2.mat"
    elif "dataset4" in FLAGS.dataset:
        d = "solid_w_points_2048.mat"
    elif "dataset5" in FLAGS.dataset:
        d = "solid_w_points_pos_err_2048.mat"
    elif "dataset6" == FLAGS.dataset:
        d = "hffh_cnn_solid_w_points_2048_training.mat"
    elif "dataset7" == FLAGS.dataset:
        d = "hffh_cnn_solid_w_points_2048_training_v2.mat"
    elif "dataset8" == FLAGS.dataset:
        d = "hffh_cnn_solid_w_points_1024_testing.mat"
    else:
        raise ValueError("Specify which dataset you want: dataset1 | dataset2 | dataset3.")
    return d

def get(dataset_dict, mat_contents, tag):
    print(tag)    
    # prints only when h5py is specified!!!
    if FLAGS.printDataInfo:
        printDataInfo(mat_contents, tag, i)
    print(f"shape: {bcolors.OKGREEN+str(mat_contents[tag].shape)+bcolors.ENDC}, type: {mat_contents[tag].dtype}")
    mat_contents[tag] = np.array(mat_contents[tag]).transpose((-1, 1, 0))
    dataset_dict[tag] = np.expand_dims(mat_contents[tag], axis=1)
    return dataset_dict

def getData(train_pct = 0.7, valid_pct = 0.15):
    """ Get new Data.
        Dataset 1: "sar1DMIMOAWR1243_256_rand.mat"
         - 32 locations with 8 virtual antennas
         - o_y = o_z = 3mm
         - no amplitude terms
        
        Dataset 2: "sar1DMIMOAWR1243_128_rand.mat"
         - 16 locations with 8 virtual antennas
         - o_y = o_z = 3mm
         - no amplitude terms

        Dataset 3: "sar1DMIMOAWR1243_128_rand_2.mat"
         - 16 locations with 8 virtual antennas
         - o_y = o_z = 3mm
         - amplitude terms

    """
    dataset = assignDataset()
    print(f'Dataset:\n{bcolors.OKGREEN+dataset+bcolors.ENDC}')
    np.random.seed(123456)
    completeDataSet = []
    
    #mat_contents = h5py.File(FLAGS.datasetPath + dataset, 'r')
    #mat_contents = hdf5storage.loadmat(FLAGS.datasetPath + dataset)
    #mat_contents = mat73.loadmat(FLAGS.datasetPath + dataset, use_attrdict = True)
    mat_contents = pymatreader.read_mat(FLAGS.datasetPath + dataset)
    #mat_contents = sio.loadmat(FLAGS.datasetPath + dataset)

    print(f" > {type(mat_contents)}")
    print(f"\n > {mat_contents.keys()}\n")
    # NOTE: when h5py is used
    #tags = list(mat_contents.keys())[2:]
    # NOTE: when pymatreader is used
    tags = list(mat_contents.keys())
    
    print(f" > {tags}")
    completeDataSet = []
    trainSet = []
    validationSet = []
    testSet = []
    dataset_dict = {}
    solidobj = False
    for i, tag in enumerate(tags):
        if tag == 'idealImageAll' or tag == 'sarImageAll': 
            dataset_dict = get(dataset_dict, mat_contents, tag)
        elif tag == 'radarImages' or tag == 'idealImages':
            dataset_dict = get(dataset_dict, mat_contents, tag)
            solidobj = True
    if solidobj:
        completeDataSet.append(dataset_dict['idealImages'])
        completeDataSet.append(dataset_dict['radarImages'])
    else:
        completeDataSet.append(dataset_dict['idealImageAll'])
        completeDataSet.append(dataset_dict['sarImageAll'])
    
    completeDataSet = np.array(completeDataSet)    
    print(f"{completeDataSet.shape}")
    
    # Standardization. Rescale data to have mean of 0 and a variance of 1.
    completeDataSet = (completeDataSet - completeDataSet.mean())/ completeDataSet.var()

    # normalize all datasets.
    #completeDataSet = normalize(torch.from_numpy(completeDataSet), p=2).numpy()

    # zip dataset along axis which has Ideal & SAR data. 
    # NOTE: Put first the SAR images and second the Ideal images
    # --- completeDataSet[0] => Ideal Images : targets
    # --- completeDataSet[1] => SAR Images : inputs
    completeDataSet = np.array(list(zip(completeDataSet[1], completeDataSet[0])))
    
    
    # shuffle data across number of samples (=pair Ideal & SAR).
    np.random.shuffle(completeDataSet[:])
    completeDataSet = completeDataSet.astype(float)
    print(f"complete Dataset : {completeDataSet.shape}")
    #completeDataSet = completeDataSet.transpose((1,0,2,3,4)).astype(float)
    
    #split data to sets {trainset, validationset, testset}
    l = completeDataSet.shape[0]
    
    trainSet = list(np.array(completeDataSet[0:int((train_pct+valid_pct)*l)]))
    validationSet = list(np.array(completeDataSet[int(train_pct*l) : int((train_pct + valid_pct)*l)]))
    testSet = list(np.array(completeDataSet[int((train_pct+valid_pct)*l):]))
    print(f"trainSet length: {bcolors.OKGREEN+str(len(trainSet))+bcolors.ENDC}, shape each: {bcolors.OKGREEN+str(trainSet[0].shape)+bcolors.ENDC}")
    print(f"validationSet length: {bcolors.OKGREEN+str(len(validationSet))+bcolors.ENDC}, shape each: {bcolors.OKGREEN+str(validationSet[0].shape)+bcolors.ENDC}")
    print(f"testSet length: {bcolors.OKGREEN+str(len(testSet))+bcolors.ENDC}, shape each: {bcolors.OKGREEN+str(testSet[0].shape)+bcolors.ENDC}")

    X = validationSet[0][0]  # get SAR images as training dataset
    Y = validationSet[0][1]  # get Ideal images as test dataset
    print(f"SAR images: {np.shape(X)}")
    print(f"Ideal images: {np.shape(Y)}")
    
    pyplot.figure()
    pyplot.contour((X[0]))
    pyplot.title (f"SAR Data")
    pyplot.savefig(f"./pics/SAR-data.jpg")

    pyplot.figure()
    pyplot.contour((Y[0]))
    pyplot.title(f"Ideal Data")
    pyplot.savefig(f"./pics/Ideal-data.jpg")
    
    return list(completeDataSet), trainSet, validationSet, testSet

def main(argv):
    del argv
    completeDataSet, trainSet, validationSet, testSet = getData(train_pct = 0.75, valid_pct = 0.125)
    

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)
