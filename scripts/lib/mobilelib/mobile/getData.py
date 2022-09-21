from absl import flags, app
from absl.flags import FLAGS
import numpy as np
from matplotlib import pyplot
import scipy.io as sio
import pymatreader
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
#from viscm import viscm
from scipy.ndimage import zoom
from mobile.utils import plot, bcolors
import logging

flags.DEFINE_string('datasetPath', 
                    '../../../data/',
                    'set the path to the dataset.')

flags.DEFINE_string('dataset', 
                    'dataset4', 
                    'set your dataset')

def assignDataset(dataset):
    if "dataset1" == dataset:
        d = "sar1DMIMOAWR1243_256_rand.mat"
    elif "dataset2" == dataset:
        d = "sar1DMIMOAWR1243_128_rand.mat"
    elif "dataset3" == dataset:
        d = "sar1DMIMOAWR1243_128_rand_2.mat"
    elif "dataset4" == dataset:
        d = "solid_w_points_2048.mat"
    elif "dataset4w0v2" == dataset:
        d = "solid_w0_points_2048_v2.mat"
    elif "dataset4wv2" == dataset:
        d = "solid_w_points_2048_v2.mat"
    elif "dataset5" == dataset:
        d = "solid_w_points_pos_err_2048.mat"
    elif "dataset6" == dataset:
        d = "hffh_cnn_solid_w_points_2048_training.mat"
    elif "dataset7" == dataset:
        d = "hffh_cnn_solid_w_points_2048_training_v2.mat"
    elif "dataset8" == dataset:
        d = "hffh_cnn_solid_w_points_1024_testing.mat"
    else:
        raise ValueError("Specify which dataset you want: dataset1 | dataset2 | dataset3.")
    return d

def get(dataset_dict, mat_contents, tag):
    logging.info(tag)
    logging.info(f"shape: {bcolors.OKGREEN+str(mat_contents[tag].shape)+bcolors.ENDC}, type: {mat_contents[tag].dtype}")
    mat_contents[tag] = np.array(mat_contents[tag]).transpose((-1, 1, 0))
    dataset_dict[tag] = np.expand_dims(mat_contents[tag], axis=1)
    return dataset_dict

def getData(train_pct = 0.7, valid_pct = 0.15, dataset="dataset8", datasetPath="../../../data/", use_real=False):

    dataset = assignDataset(dataset=dataset)
    logging.info(f'Dataset: {bcolors.OKGREEN+dataset+bcolors.ENDC}')
    np.random.seed(123456)
    completeDataSet = []

    mat_contents = pymatreader.read_mat(datasetPath + dataset)
    if dataset == "dataset6":
        dataset = "dataset7"
        dataset2 = assignDataset(dataset=dataset)
        mat_contents2 = pymatreader.read_mat(datasetPath + dataset2)
        mat_contents["radarImagesFFH"] = np.concatenate([mat_contents["radarImagesFFH"], mat_contents2["radarImagesFFH"]], axis=2)
        mat_contents["idealImages"] = np.concatenate([mat_contents["idealImages"], mat_contents2["idealImages"]], axis=2)

    logging.info(f" {type(mat_contents)}")
    logging.info(f" {mat_contents.keys()}")
    # NOTE: when h5py is used
    #tags = list(mat_contents.keys())[2:]
    # NOTE: when pymatreader is used
    tags = list(mat_contents.keys())
    
    logging.info(f" {tags}")
    completeDataSet = []
    trainSet = []
    validationSet = []
    testSet = []
    dataset_dict = {}
    solidobj = False
    for i, tag in enumerate(tags):
        if tag == 'idealImageAll' or tag == 'sarImageAll': 
            dataset_dict = get(dataset_dict, mat_contents, tag)
        elif tag == 'radarImages' or tag == 'idealImages' or tag == 'radarImagesFFH':
            dataset_dict = get(dataset_dict, mat_contents, tag)
            if 'radarImages' in dataset_dict.keys():
                dataset_dict['radarImages'] = np.abs(dataset_dict['radarImages'])
            elif 'radarImagesFFH' in dataset_dict.keys():
                dataset_dict['radarImagesFFH'] = np.abs(dataset_dict['radarImagesFFH'])
            solidobj = True
    if solidobj:
        completeDataSet.append(dataset_dict['idealImages'])
        if 'radarImages' in dataset_dict.keys():
            completeDataSet.append(dataset_dict['radarImages'])
        elif 'radarImagesFFH' in dataset_dict.keys():
            completeDataSet.append(dataset_dict['radarImagesFFH'])
    else:
        completeDataSet.append(dataset_dict['idealImageAll'])
        completeDataSet.append(dataset_dict['sarImageAll'])
    
    completeDataSet = np.array(completeDataSet).astype(float)
    logging.info(f"{completeDataSet.shape}")
    syn_distorted = completeDataSet[1]
    syn_ideal = completeDataSet[0]

    if use_real:
        ##############################################################################
        # NOTE: Distorted:  [synthetic distorted + real distorted]                   #
        # ------ synthetic distorted:  2048 samples  ->  index range:    0-2047      #
        # ------ real distorted     :    45 samples  ->  index range: 2048-2092      #
        # ________________________________________________________________________   #
        # ------ Total:                2093 samples  ->  index range:    0-2092      #
        # ________________________________________________________________________   #
        # NOTE: Distorted + Ideal -> Semi-Supervised Learning                                                    #
        # ------ distorted          :  2093 samples  ->  index range:    0-2092      #
        # ------ synthetic Ideal    :  2048 samples  ->  index range:    0-2047      #
        ##############################################################################

        # Specify to i, which data to load.
        # 1: old, 2: new
        i = 2
        d = lambda x: f"real_data_xy_images_{x}.mat"
        x = sio.loadmat(datasetPath+d(i))
        # fix resolution
        if i == 1: # old dataset
            real_distorted = zoom(np.array(x['data']), (1, 2, 2)).transpose((0,2,1))
        elif i == 2: # new dataset
            real_distorted = np.array(x['data']).transpose((2,1,0))

        # fix shape of data
        real_distorted = (real_distorted - real_distorted.mean()) / real_distorted.std()
        real_distorted = np.expand_dims(real_distorted, axis=1)
        l = real_distorted.shape[0]
        train_real_distorted = np.array(real_distorted[:int(0.8*l)])
        test_real_distorted = np.array(real_distorted[int(0.8*l):])
        logging.info(f'Real`s train:\n{bcolors.OKGREEN + str(train_real_distorted.shape) + bcolors.ENDC}')
        logging.info(f'Real`s test:\n{bcolors.OKGREEN + str(test_real_distorted.shape) + bcolors.ENDC}')
        syn_data = np.concatenate((syn_distorted, syn_ideal), axis=0)
        # normalize synth data
        syn_data = (syn_data - syn_data.mean(axis=0)) / syn_data.std(axis=0)
        # distinguish synth distorted & ideal
        syn_distorted, syn_ideal = syn_data[:2048], syn_data[2048:2*2048]
        # collect all distorted data together
        distorted_data = np.concatenate((syn_distorted, real_distorted), axis=0)
        return distorted_data, syn_ideal, train_real_distorted, test_real_distorted
    else:
        # Standardization. Rescale data to have mean of 0 and a standard deviation of 1.
        completeDataSet = (completeDataSet - completeDataSet.mean())/completeDataSet.std()
        completeDataSet = completeDataSet - completeDataSet.min()

        # zip dataset along axis which has Ideal & SAR data.
        # NOTE: Put first the SAR images and second the Ideal images
        # --- completeDataSet[0] => Ideal Images : targets
        # --- completeDataSet[1] => SAR Images : inputs
        completeDataSet = np.array(list(zip(completeDataSet[1], completeDataSet[0])))

        # shuffle data across number of pairs (=pair of Ideal & SAR).
        #np.random.shuffle(completeDataSet[:])
        logging.info(f"complete Dataset : {completeDataSet.shape}")
        #completeDataSet = completeDataSet.transpose((1,0,2,3,4)).astype(float)

        #split data to sets {trainset, validationset, testset}
        l = completeDataSet.shape[0]

        trainSet = list(np.array(completeDataSet[0:int((train_pct+valid_pct)*l)]))
        validationSet = list(np.array(completeDataSet[int(train_pct*l) : int((train_pct + valid_pct)*l)]))
        testSet = list(np.array(completeDataSet[int((train_pct+valid_pct)*l):]))
        logging.info(f"trainSet length: {bcolors.OKGREEN+str(len(trainSet))+bcolors.ENDC}, shape each: {bcolors.OKGREEN+str(trainSet[0].shape)+bcolors.ENDC}")
        logging.info(f"validationSet length: {bcolors.OKGREEN+str(len(validationSet))+bcolors.ENDC}, shape each: {bcolors.OKGREEN+str(validationSet[0].shape)+bcolors.ENDC}")
        logging.info(f"testSet length: {bcolors.OKGREEN+str(len(testSet))+bcolors.ENDC}, shape each: {bcolors.OKGREEN+str(testSet[0].shape)+bcolors.ENDC}")

        X = trainSet[0][0]  # get SAR images as training dataset
        Y = trainSet[0][1]  # get Ideal images as test dataset
        logging.info(f"SAR images: {np.shape(X)}")
        logging.info(f"Ideal images: {np.shape(Y)}")

        return list(completeDataSet), trainSet, validationSet, testSet

def main(argv):
    del argv
    completeDataSet, trainSet, validationSet, testSet = getData(train_pct = 0.75, valid_pct = 0.125)

