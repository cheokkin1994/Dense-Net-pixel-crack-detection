import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
from skimage.util.shape import view_as_windows
import pickle
import datetime
from sklearn.feature_extraction import image


# Patches extraction, also return a list indicating whether a patch center is cracked.
def patch_extraction(img, gt ,size=27):
    labels = []
    h_size = img.shape[0] - size + 1
    w_size = img.shape[1] - size + 1

    # Calculate the image center pixel coordinates
    for h in range(h_size):
        for w in range(w_size):
            center = gt[h+ size//2, w + size//2]
            if center == 255:
                labels.append(False)
            else:
                labels.append(True)     
    # Extract pactches  
    image_patches = image.extract_patches_2d(img, patch_size=(size, size))
    gt_patches = image.extract_patches_2d(gt, patch_size=(size, size))
    return image_patches, gt_patches, labels


# Function to generate positive or negative samples, and their corresponding labels
def generative_samples (img, gt , size= 27, gt_size = 5,sample_size = 5, positive = True):
    image_patches, gt_patches, labels = patch_extraction(img,gt)
    output_imgs = []
    output_labels = []
    center = size // 2
    if positive:
        # find labels with True
        p_index = np.array([i for i, v in enumerate(labels) if v])
        # check empty
        if len(p_index) == 0:
            return output_imgs,output_labels
        
        if sample_size =='all' or sample_size > len(p_index):
            sample_size = len(p_index)
        rnd_index = p_index[np.random.choice(len(p_index), sample_size, replace=False)]
        
        for i in rnd_index:
            output_imgs.append(image_patches[i])
            output_labels.append(gt_patches[i][center-gt_size//2: center + gt_size//2 +1,
                                               center-gt_size//2: center + gt_size//2 +1])
    else:
        p_index = np.array([i for i, v in enumerate(labels) if not v])
         # check empty
        if len(p_index) == 0:
            return output_imgs,output_labels
        
        if sample_size =='all' or sample_size > len(p_index):
            sample_size = len(p_index)
        
        rnd_index = p_index[np.random.choice(p_index.shape[0], sample_size, replace=False)]
        for i in rnd_index:
            output_imgs.append(image_patches[i])
            output_labels.append(gt_patches[i][center-gt_size//2: center + gt_size//2 +1,
                                               center-gt_size//2: center + gt_size//2 +1])
            
    return output_imgs, output_labels

# Function to scale data to [-1, 1]
def convert_data(data):
    data = 2 * (np.array(data, dtype="float") / 255.0) - 1
    return np.expand_dims(data,-1)

# Function to convert pixels to 1 and 0, 1 indicates a crack, 0 indicates non-cracked. The raw ground truth provided in the dataset uses 0 in gray-scale to denote a cracked pixel and 255 for non-cracked pixels. 
def convert_label (label):
    label = label / 255
    label = label  + (-1) ** (label/1 +2) 
    label = np.reshape(label, (label.shape[0], 25))
    return label


def main():
	# Step 1, load all images and ground truths
	# Reading raw images
	# Loading images 
	np.random.seed(1)
	filename = r'IMAGES\AIGLE_RN'
	images_dict = {}
	gt_dict = {}

	for filename in glob.glob(r'IMAGES\AIGLE_RN\Im_GT*.png'):
	    # Reading files, scaling to -1 and 1
	    pic = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	    filename = filename.replace(r'IMAGES\AIGLE_RN\Im_GT_','')
	    filename = filename.replace('or.png','')
	    images_dict[filename] = pic

	print(images_dict.keys(), 'Pictures are loaded into dictionary')
	print('-----------------------')
	# Reading ground truth
	for filename in glob.glob(r'GROUND_TRUTH\AIGLE_RN\GT*.png'):\
	    # Reading files, scaling to -1 and 1
	    pic = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	    filename = filename.replace(r'GROUND_TRUTH\AIGLE_RN\GT_','')
	    filename = filename.replace('.png','')
	    gt_dict[filename] = pic

	print(gt_dict.keys(), 'Ground truth are loaded into dictionary')


	# Step2：Divide the images into training and testing set
	# n defines the number of images to be used to generate training, total images in the paper is 24
	n = 24 
	# output positive samples
	training_set = (np.random.choice(list(images_dict.keys()), n, replace=False))
	testing_set = np.array( [x for x in list(images_dict.keys()) if x not in training_set])
	
	# After splitting the images set, extract patches from them 
	positive_train, positive_labels= [],[]
	negative_train, negative_labels = [],[]
	for sample in training_set:
	    print(sample, 'positive samples is extracting', datetime.datetime.now())
	    img = images_dict [sample]
	    gt = gt_dict[sample]
	    #  positive samples
	    list_of_sample, list_of_labels = generative_samples(img,gt,sample_size='all', positive=True)
	    print('The number of positive patch in this pic:',len(list_of_sample))
	    if len(list_of_sample) > 0:
	        for i in list_of_sample:
	                positive_train.append(i)
	        for i in list_of_labels:
	                positive_labels.append(i)

	    # negative samples
	    print(sample, 'negative samples is extracting', datetime.datetime.now())
	    list_of_sample, list_of_labels = generative_samples(img,gt,sample_size='all', positive=False)
	    print('The number of negative patch in this pic:', len(list_of_sample))
	    for i in list_of_sample:
	        negative_train.append(i)
	    for i in list_of_labels:
	        negative_labels.append(i)
	    del list_of_sample, list_of_labels
	    print('----------------------------')
	print('Total patches:' ,str(len(positive_labels) + len(negative_labels)))

	# Step 3: Control the ratio between positive and negative samples
	total_samples = len(positive_labels) + len(negative_labels) 
	# we set up negative samples is 3 times more than positive samples
	ratio = 3
	positive_n = len(positive_labels)
	negative_n = positive_n  * ratio
	# Random selection
	positive_selected = np.random.choice(len(positive_train),positive_n,False)
	negative_selected = np.random.choice(len(negative_train),negative_n,False)
	print('The total number of patch extracted:', total_samples)
	print('The number of positive samples:', len(positive_selected))
	print('The number of negative samples:', len(negative_selected))
	print('Ratio: ' ,len(negative_selected) / len(positive_selected))	

	print('Preparing validation and training data....')
	#  Step 4: Obtain 20% from the data as the validation set for hyperparameter selection.
	validation_ratio = 0.2
	validation_selected_p =  np.random.choice(positive_selected,round((positive_n) * validation_ratio),False)
	validation_selected_n = np.random.choice( negative_selected, len(validation_selected_p) * ratio  ,False)
	validation_data = np.concatenate([np.array(positive_train)[validation_selected_p],np.array(negative_train)[validation_selected_n] ], axis=0)
	validation_label = np.concatenate([np.array(positive_labels)[validation_selected_p], np.array(negative_labels)[validation_selected_n] ], axis=0)

	# Step 5: Now put data together, shuffle, and scale
	positive_set = np.array([x for x in positive_selected if x not in validation_selected_p])
	negative_set = np.array([x for x in negative_selected if x not in validation_selected_n])
	train_data = np.concatenate([np.array(positive_train)[positive_set],np.array(negative_train)[negative_set] ], axis=0)
	label = np.concatenate([np.array(positive_labels)[positive_set], np.array(negative_labels)[negative_set]],	axis=0)

	# Scaling to 255 to -1 to 1, label is 1 if crack pixel
	train_data = convert_data(train_data)
	label = convert_label(label)
	validation_data = convert_data(validation_data)
	validation_label = convert_label(validation_label)
	# Shuffling
	N = np.random.choice(range(train_data.shape[0]), train_data.shape[0], replace=False)
	train_data = train_data[N]
	label = label[N]
	
	# output the training and vlidation data in pickle
	with open('train_valid.pickle', 'wb') as f:
		pickle.dump([train_data, label, validation_data, validation_label], f)
    # Also output the testing data for evaluation 
	names = []
	testdata = []
	testlabel = []
	for i in testing_set:
	    names.append(i)
	    testdata.append(images_dict[i])
	    testlabel.append(gt_dict[i])
	with open('testing.pickle', 'wb') as f:
	    pickle.dump([names, testdata, testlabel], f)

if __name__== "__main__":
  main()