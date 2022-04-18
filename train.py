''' @ author: Mohammadreza Baghery '''

import numpy as np 
from matplotlib import pyplot as plt 
import cv2
import pandas as pd
from skimage.filters import sobel, roberts, scharr, prewitt
from scipy import ndimage as nd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from skimage.filters.rank import entropy 
from skimage.morphology import disk 
import pickle
import os
import glob


###########################################################################
## STEP 1: READ TRAINING IMAGES AND EXTRACT FEATURES
###########################################################################

img_path = ''

mask_path = ''

image_dataset = pd.DataFrame() # Dataframe to capture image features
 
for image in os.listdir(img_path):
    if 'tif' in str(image):
        print(image)
        df = pd.DataFrame()
        input_img = cv2.imread(img_path + image)

        # check if the input image is RGB or gray and convert to grey if RGB
        if input_img.ndim == 3 and input_img.shape[-1] == 3:
            img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        elif input_img.ndim == 2:
            img = input_img
        else: 
            raise Exception('The module works only with grayscale and RGB images!')
 
        print(img.shape)

        ###########################################################################
        # SRART ADDING DATA TO THE DATAFRAME

        # Add pixel values to the dataframe 
        pixel_values = img.reshape(-1) # Unwrapping the 2-D image into a 1 dimensional vector
        df['Pixel_Value'] = pixel_values   # pixel value itself as a feature
        df['Image_Name'] = image           # capture image name as we read multiple images 

        ###########################################################################

        # Adding gabor features 
        Gab_num = 1
        for theta in range(2):
            theta = theta/4 * np.pi 
            for sigma in (1, 3): # sigma with 1 and 3
                for lamda in np.arange(0, np.pi, np.pi/4):
                    for gamma in (0.05, 0.5):
                        gabor_label = 'Gabor ' + str(Gab_num)
                        kernel = cv2.getGaborKernel((5, 5), sigma, theta, lamda, gamma, 0, ktype = cv2.CV_32F)
                        fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                        filter_img = fimg.reshape(-1)  # converting 2D to 1D array
                        df[gabor_label] = filter_img
                        Gab_num += 1


        # Adding Canny edge feature 
        edge_canny = cv2.Canny(img, 100, 200)
        edge_canny_reshp = edge_canny.reshape(-1) #unwrapping the 2-D image 
        df ['Canny Edge'] = edge_canny_reshp

        # Adding Sobel edge feature 
        edge_sobel = sobel(img)
        edge_sobel_reshp  = edge_sobel.reshape(-1) #unwrapping the 2-D image 
        df['Sobel Edge'] = edge_sobel_reshp

        # Adding Sobel Roberts edge feature 
        edge_roberts = roberts(img)
        edge_roberts_reshp  = edge_roberts.reshape(-1) #unwrapping the 2-D image 
        df['Roberts Edge'] = edge_roberts_reshp

        # Adding Sobel Scharr edge feature 
        edge_scharr = scharr(img)
        edge_scharr_reshp  = edge_scharr.reshape(-1) #unwrapping the 2-D image 
        df['Scharr Edge'] = edge_scharr_reshp

        # Adding Sobel Prewitt edge feature 
        edge_prewitt = prewitt(img)
        edge_prewitt_reshp  = edge_prewitt.reshape(-1) #unwrapping the 2-D image 
        df['Prewitt Edge'] = edge_prewitt_reshp

        #entropy = entropy(img, disk(1))
        #entropy_reshp = entropy.reshape(-1)
        #df['Entropy'] = entropy_reshp

        img_color = cv2.imread(img_path + image)
        denois_num = 9
        for den in range(30):
            dst = cv2.fastNlMeansDenoisingColored(img_color, None, denois_num, denois_num, 7, 21)
            dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            dst_gray_reshp = dst_gray.reshape(-1) #unwrapping the 2-D image
            df['Denoised h' + str(denois_num)] = dst_gray_reshp
            denois_num += 1

        ### Adding other features, Gaussian, Mediian, Variance
        # Adding Gaussian features 
        Gauss_num = 1
        for gauss in range(10):
            gaussian = nd.gaussian_filter(img, sigma = Gauss_num)
            gaussian_reshp = gaussian.reshape(-1) #unwrapping the 2-D image
            df['Gaussian s' + str(Gauss_num)] = gaussian_reshp
            Gauss_num += 1

        # Adding blur features
        blur_num = 1
        for blur in range (15):
            blur = cv2.blur(img, (blur_num, blur_num))
            blur_reshp = blur.reshape(-1) #unwrapping the 2-D image
            df['Blur s' + str(blur_num)] = blur_reshp

        # Adding Median features 
        Median_num = 1
        for med in range(10):
            median = nd.median_filter(img, size = Median_num)
            median_reshp = median.reshape(-1) #unwrapping the 2-D image
            df['Median s' + str(Median_num)] = median_reshp
            Median_num += 1

        # Adding Variance features 
        var_num = 1
        for med in range(10):
            variance = nd.generic_filter(img, np.var, size = var_num)
            variance_reshp = variance.reshape(-1) #unwrapping the 2-D image
            df['Variance s' + str(var_num)] = variance_reshp
            var_num += 1

########################################################################################
    # Update dataframe for images to include details for each image in the loop 
        image_dataset = image_dataset.append(df)

########################################################################################

###########################################################################
## STEP 2: READ LABELED IMAGES (MASKS) AND CREATE ANOTHER DATAFRAME WITH
#          LABEL VALUES AND LABEL FILE NAMES
###########################################################################

mask_dataset = pd.DataFrame() # Create dataframe to capture mask info. 

for mask in os.listdir(mask_path):  # iterate through each file to perform some action

    if 'tif' in str(mask):
        print(mask)

        df2 = pd.DataFrame() 
        input_mask = cv2.imread(mask_path + mask)

        # check if the input mask is RGB or gray and convert to grey if RGB
        if input_mask.ndim == 3 and input_mask.shape[-1] == 3:
            label = cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY)
        elif input_mask.ndim == 2:
            label = input_mask
        else: 
            raise Exception('The module works only with grayscale and RGB images!')

        # Add pixel values to the dataframe 
        label_values = label.reshape(-1)
        df2['Label_Value'] = label_values
        df2['Mask_Name'] = mask 

        mask_dataset = mask_dataset.append(df2)
        print(label.shape)

###########################################################################
## STEP 3: GET DATA READY FOR RANDOM FOREST (OR OTHER CASSIFIER)
#          COMBINE BOTH DATAFRAMES INTO A SINGLE DATASET
###########################################################################


print(image_dataset.shape)

print(mask_dataset.shape)


### INJA @@@ ###
dataset = pd.concat([image_dataset, mask_dataset], axis = 1) # Concatenate both image and mask datasets 

dataset = dataset[dataset.Label_Value != 0] # droping pixels with value of 0 

#Assing training features to X and labels to Y 
# drop columns that are not relevant for training (non-features)
X = dataset.drop(labels = ['Image_Name', 'Mask_Name', 'Label_Value'], axis = 1)

#Assign label values to Y (our prediction)
Y = dataset['Label_Value'].values

##split data into train and test to verify accuracy after fitting the model 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 20)

###########################################################################
## STEP 4: Define the classfier and fit a model with our training data
###########################################################################

# Import ML algorithm and train the model
model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# traing the model on training data 
model.fit(X_train, Y_train)


###########################################################################
## STEP 5: Accuracy check AND CHECKING THE IMPORTANCE OF EACH FEATURE
###########################################################################

# Calculate accuracy on test data 
prediction_test = model.predict(X_test)

print('Accuracy = ', metrics.accuracy_score(Y_test, prediction_test))  

importances = list(model.feature_importances_)
features_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index = features_list).sort_values(ascending = False)

print(feature_imp)


###########################################################################
## STEP 5: SAVE MODEL FOR FUTURE USE 
###########################################################################

# save the train model as pickle string for future use 
filename = 'TH neurons model 2.0'

pickle.dump(model, open(filename, 'wb' ))

load_model = pickle.load(open(filename, 'rb'))


#result = load_model.predict(X)
#segmented = result.reshape((img.shape))
#plt.imsave('/Users/MohReza/Desktop/PhD/TH count/image/TH neurons model_segmented_revised_04.tiff', segmented, cmap = 'jet')



















