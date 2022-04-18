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
import glob
from skimage import measure 
from skimage import morphology
from skimage.feature import canny 
from skimage import img_as_float
from skimage.color import rgb2gray, label2rgb


input_path = ''
output_path = ''

# binary threshold for the train image (mask image); user spcecies the value after observing the train image
threshold = 0 

def feature_extraction(img):
    df = pd.DataFrame()

    # Add pixel values to the dataframe 
    pixel_values = img.reshape(-1) # Unwrapping the 2-D image into a 1 dimensional vector
    df['Pixel_Value'] = pixel_values   # pixel value itself as a feature

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

    img_color = cv2.imread(input_path)
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

    return df

#filename = 'TH neurons model'
#load_model = pickle.load(open(filename, 'rb'))

filename = '' # the image developed as our model in training module

load_model = pickle.load(open(filename, 'rb'))

img1 = cv2.imread(input_path)

img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

X = feature_extraction(img)
result = load_model.predict(X)
segmented = result.reshape((img.shape))


label_image = measure.label(segmented > threshold)

mask = morphology.remove_small_objects(label_image, 60)

plt.imshow(mask, cmap = 'jet')
plt.show()


image_label_overlay = label2rgb(mask, image=img1, bg_label=0)

original_image_gray = rgb2gray(img1)

# defining the properites of the segmented image
props = measure.regionprops_table(mask, original_image_gray, properties = ['label', 'area', 'centroid', 'mean_intensity', 'eccentricity'])

# converting the defined properites into pandas dataframe
df_cells = pd.DataFrame(props)

label = df_cells['label']

x = df_cells['centroid-1'].tolist()
y = df_cells['centroid-0'].tolist()

x_integ = [i for i in x x_integ.append(int(i))]
y_integ = [i for i in y y_integ.append(int(i))]

coordinates = list(zip(x_integ, y_integ))

for i in coordinates:
    image_coordin = cv2.circle(img1, i, 12, (255, 0, 0), 2)


plt.imsave(output_path + 'coordinates.tiff', image_coordin)
plt.imsave(output_path + 'overlay.tiff', image_label_overlay)

#plt.imshow(image_coordin)
#plt.show()
