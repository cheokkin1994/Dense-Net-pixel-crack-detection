import numpy as np
import glob
import cv2
import datetime
from skimage.util.shape import view_as_windows
import pickle
import re
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os


# Generate test patches with symmetric padding for detection near the edge
def generate_test_patches (test_image, patch_size = 27):
    padwidth = patch_size //2 
    # Pad for boundary issue
    pad_img = np.pad(test_image,padwidth,'symmetric')
    patches = view_as_windows(pad_img, (patch_size,patch_size))
    patches = patches.reshape(patches.shape[0] * patches.shape[1] , 27, 27)
    return (patches)

# Auxilary function for crack detection, summming up the matrix at particular position
def addAtPos(mat1, mat2, xypos):
    x, y = xypos
    ysize, xsize = mat2.shape
    xmax, ymax = (x + xsize), (y + ysize)
    mat1[y:ymax, x:xmax] += mat2 
    mat1[y:ymax, x:xmax] = mat1[y:ymax, x:xmax]
    return mat1


# function to detect cracks, threshold as 0.5, returns a binary output picture 
def detect_crack(model,img, size=5, threshold = 0.5):
    img = 2 * (np.array(img, dtype="float") / 255.0) - 1
    patches = generate_test_patches(img)
    votes = [0] * len(patches)
    # Zeros are used to store the probability map 
    zeros = np.zeros(img.shape)
    zeros = np.pad(zeros,size//2, 'constant')
    # Masked ones to calculate how many decisions for the pixel
    ones = np.ones(img.shape)
    ones = np.pad(ones, size//2, 'constant')
    print('Number of prediction:', img.shape[0] * img.shape[1])
    r, c = img.shape[0], img.shape[1]
    for i in range(r):
        for j in range(c):
            patch = patches[i * img.shape[1] + j] 
            votes[i * img.shape[1] + j] = (size **2) - np.sum(ones[i:i+size, j:j+size] == 0)
            o = model.predict(np.expand_dims(img_to_array(patch), axis=0)).reshape(5,5)
            zeros = addAtPos(zeros, o, (j,i))
    # Thresholding
    zeros = zeros[(size//2):-(size//2), (size//2):-(size//2)] # Cut the boundary
    votes = np.array(votes).reshape(zeros.shape)
    zeros = zeros/ votes
    zeros [ zeros <= threshold] = 0
    zeros [ zeros > threshold] = -255
    zeros += 255
    return zeros


#  The function counts the true postive, false positive, false negative and true negative. For TP, accept 2 pixel distances.
def evaluation (img, gt, d =2):
  	#  First padd extra useless values of d with to ease comparison.
    img = np.pad(img, d, 'constant', constant_values=(99, 99))
    gt = np.pad(gt, d, 'constant', constant_values=(99, 99))
    cracks, non_cracks = [], []

    # initialize the counts
    tp = 0
    fp = 0 
    fn = 0
    tn = 0
    # First divide the pixels to cracked and non cracked for the predictions
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r,c] == 0:
                cracks.append((r,c))
            elif img[r,c] == 255:
                non_cracks.append((r,c))
     # Counting tp and fp            
    for point in cracks:
        row = point[0]
        col = point[1]
        # d pixels horizontally or vertically from the target, or the square with a side of d - 1, centered at the target pixel 
        values = (gt[row - d  +1 : row + d , col - d +1: col + d ]).ravel()
        square_pixels = np.array([gt[row -d,col], gt[row + d,col], gt[row ,col-d], gt[row, col +d]])
        values = (np.append(values, square_pixels))
        if ( 0 in values):
            tp += 1
        else:
            fp +=1
    # counting fn and tn
    for point in non_cracks:
        row = point[0]
        col = point[1]
        if gt[row,col] == 0:
            fn +=1
        else:
            tn +=1
    # Calculations 
    if tp == 0 or (tp + fp) == 0 or (tp + fn) == 0:
        print('local F1:', 'Zero precision and recall')
    else:
    	# adjusted precision and recall
        pr = tp/ (tp + fp)
        re =  tp / (tp + fn)
        print('local F1:', 2 * pr * re / (pr+re) )
        print('Adjusted precision:',pr)
        print('local Recall:', re)

    print('tp', tp)
    print('fp',fp)
    print('fn',fn)
    print('tn', tn)
    print('\n')
    return tp, fp, fn, tn

# Evalaution the proposed model
def evaluate_proposed (names, testing_imgs, testing_labels, model):
    print('Number of images to be evaluated', len(names))
    all_tp, all_fp, all_fn = [], [], []
    out = []
    # For each image count the tp, fp and fn 
    for i in range(len(names)):
        img = testing_imgs[i]
        name = names[i]
        label = testing_labels[i]
        print('The result for image', name, datetime.datetime.now())
        prediction = detect_crack(model,img)
        out.append(prediction)
        tp, fp, fn , tn= evaluation( prediction, label)
        all_tp.append(tp)
        all_fp.append(fp)
        all_fn.append(fn)
    # sum all as a overall evaluation 
    sum_tp = np.sum(all_tp)
    sum_fp = np.sum(all_fp)
    sum_fn = np.sum(all_fn)
    precision = sum_tp / (sum_tp + sum_fp)
    recall = sum_tp/ (sum_tp + sum_fn)
    return precision, recall, 2* precision * recall / (precision+recall), out

# function to evaluate other methods based on the images provided in the dataset
def evaluate_others (names, testingimgs, test_labels):
    print('Number of images to be evaluated', len(names))
    result = dict()
    j = 0
    for i in range(len(names)):
        print('Evaluating', names[i])
        name = 'Res_GT_' + names[i] + '*'
        path = os.path.join(r'RESULTS\AIGLE_RN',name)
        imgs_list = (glob.glob(path))
        for img_path in imgs_list:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            method = (re.search(r'(?<=_)\w{2,3}(?=\.png)',img_path).group())
            print('Result of Method:', method)
            tp,fp, fn, tn = (evaluation(img, test_labels[i]))
            if method not in result:
                result[method] = [ [] for x in range(3)]
            result[method][0].append(tp)
            result[method][1].append(fp)
            result[method][2].append(fn)
        print('-------------------------------------')

    measures = dict()
    for k, v in result.items():
        sum_tp = np.sum(v[0])
        sum_fp = np.sum(v[1])
        sum_fn = np.sum(v[2])
        precision = sum_tp / (sum_tp + sum_fp)
        recall = sum_tp/ (sum_tp + sum_fn)
        measures[k] = [precision ,recall , 2* precision * recall / (precision+recall)]
    return result, measures

def main():
	# Open model, testing file for evaluation
    model = load_model('denseModel.h5')
    with open('testing.pickle', 'rb') as f:
        names, testing_imgs, testing_labels = pickle.load(f)
    result = evaluate_proposed(names, testing_imgs,testing_labels, model)
    other_result = evaluate_others(names, testing_imgs, testing_labels)
    print(other_result)
    with open('result.txt', 'w') as f:
        f.write(str('Proposed method (PR, RE, F1) :\n'))
        f.write(str('Precision : ' + str(result[0]) +'\n'))
        f.write(str('Recall: ' + str(result[1]) + '\n'))
        f.write(str('F1: ' + str(result[2]) +'\n'))
        f.write('\n')
        f.write(str('Other methods (PR, RE, F1) :\n'))
        f.write(str(other_result[1]))
        print('Precision:', result[0])
        print('Recall:', result[1])
        print('F1:', result[2])
	# save result
    if not os.path.exists('prediction'):
    	os.makedirs('prediction')

    for i in range(len(result[3])):
	    n = names[i]
	    cv2.imwrite(( 'prediction/' + n + ".png"),result[3][i])

if __name__== "__main__":
  main()
