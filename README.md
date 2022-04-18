# Random Forest Segmentation

## A supervised machine learning algorithm for cell detection

Using Random Forest segmentation, which is a supervised machine learning algorithm, 
the module segments cell bodies in noisy microscipic images. Initially, cell bodies 
images are labeled in a few images. These labeled images constitute the training dataset. Then using the 
"training module", we extract a variety of features (e.g Gabor filter, Sobel filter, Gaussian filter, ...) from the
training images. These features are then fed to a machine learning algorithm (Random Forest). The algorithm developes
a trained model. The trained model is then validated using a a few unlabled images (as the test dataset). 

Once the model validation is satisfactory, the model is saved as a pickle file. Finally, using the "prediction module", we
can segment cell bodies in other images in our dataset. 
