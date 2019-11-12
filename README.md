## EVALUATE TECHNIQUES FOR WIFI LOCATIONING
### Introduction
GPS based location estimation is quite popular. Mobile device, smart-phone, i-phone are utilizing GPS receivers for location estimation. It, however, does not work for indoor environments because of the fact that GPS satellite signals cannot be received in indoor environments. Therefore, automatic user localization has been a hot research topic in the last years. Automatic user localization consists of estimating the position of the user (latitude, longitude and altitude) by using an electronic device, usually a mobile phone. The approach covered in this project known as fingerprinting uses a training set of particular positions in a building and the corresponding signal strength from any and all WAP’s in the vicinity. In this way, a ‘fingerprint’ of WAP signal strength is produced for each position in a building. One challenge to this approach is that the received signal strength can vary based on the phone brand and model, and the position of the phone (i.e. the height of phone owner).

### Objective

To investigate classification & regression models to predict the location (building, floor, latitudes and longitudes) on the multi-building of Jaume University in Spain.
### General Overview of the Data

•	The database consists of 19937 training/reference records which were generated with an Android application called CaptureLoc.
•	1111 validation/test records which were generated with another Android application called ValidateLoc.
•	The 529 attributes contain the WiFi fingerprint, the coordinates where it was taken, and other useful information. 

•	The intensity values are represented as negative integer values ranging -104dBm (extremely poor signal) to 0dBm.
•	The positive value 100 is used to denote when a WAP was not detected.

### Preprocessing of the Data


### Intial Exploration of the Data
We can also visualize how the number of WAPs detected varies across the 3 buildings and across floors. The boxplot below shows the distribution of WAPs across the buildings. Building 3 has the highest median detected WAPs whereas Building 0 and 1 appear to have similar medians. The distribution in building 1 also reaches to the lower end of WAPs detected relative to the other buildings.

We can use the latitude and longitude data in the data set to plot the location reference points of each building by floor

### Building Predictive Models
After cleaning and visualization of the data removed the attributes that won’t be needed in a predictive model. Used the train Control function to setup many parameters used in the model fitting. The method argument sets the resampling technique which in this case is repeated cross validation (CV) or repeated cv. The number argument sets the number of folds in the k-fold-CV which will then be repeated 3 times. The allow Parallel argument allows to take advantage of R’s parallel processing capabilities. The final parameter allows us to reproducibly set the seed for work going on during parallel processing.

The KNN, C50, Random forest and svm models were used for the classification. For the regression the models used were KNN, Random forest and SVM.

### Comparison of the models
The results of our predictive models were compared by using the resamples () function followed by bwplot.

#### Building ID
#### Floor
#### Longitude
#### Latitude

### Train Final Predictive Model
Looking at the model output we can see that the KNN algorithms provided a good performance compared with other classifiers approaches such as SVM, Random Forest and C5.0. The following are the accuracies for the model.

### Assess the model

### Summary

In summary, we’ve used the UJIIndoorLoc data set of Wi-Fi fingerprints to train and compare several classification models. Based on the kind of prediction (building, floor or latitude/longitude), the best variables for building these predictions change. The model has an error of 8.39 m predicting longitude and predicting latitude.








