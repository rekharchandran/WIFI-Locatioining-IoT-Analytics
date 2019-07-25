###Loading Libraries
library(readr)
library(tidyverse)
library(Hmisc)
library(VIM)
library(dplyr)
library(ggplot2)
library(plotly)
library(rgl)         
library(scatterplot3d) 
library(scales)      
library(kknn) 
library(caret)
library(ISLR)
library(gridExtra)
library(grid)
library(reshape2)
library(scales)
library(class)
library(C50)
library(kernlab) 
library(ranger)      
library(C50)         
library(e1071)
library(MASS)
library(klaR)
library(RColorBrewer)
library(randomForest)
library(data.table)

## uploading the data set
#trainingset
setwd("Documents/WIFI Task/UJIndoorLoc/")
getwd()
trainingData <- read.csv("trainingData.csv", na = '100')# training set
#uploading the validation dataset
validationData <- read.csv("validationData.csv",na = '100')# validation data

####Inspecting the data set
dim(trainingData)
str(trainingData)
attributes(trainingData)
summary(trainingData)
dim(validationData)
summary(validationData)
####Process the Data
trainingData <- sapply(trainingData, as.numeric)
trainingData <- as_tibble(trainingData)
validationData <- sapply(validationData, as.numeric)
validationData <- as_tibble(validationData)
str(trainingData)
str(validationData)

#-Convert features to categorical training data
trainingData$FLOOR <- as.factor(trainingData$FLOOR)
trainingData$BUILDINGID <- as.factor(trainingData$BUILDINGID)
trainingData$SPACEID <- as.factor(trainingData$SPACEID)
trainingData$RELATIVEPOSITION <- as.factor(trainingData$RELATIVEPOSITION)

#-Convert features to categorical validation data

validationData$FLOOR <- as.factor(validationData$FLOOR)
validationData$BUILDINGID <- as.factor(validationData$BUILDINGID)
validationData$SPACEID <- as.factor(validationData$SPACEID)
validationData$RELATIVEPOSITION <- as.factor(validationData$RELATIVEPOSITION)

# #-Visualize extent and pattern of NA values
#aggr(trainingData, col=c('navyblue','red'),
#     numbers=TRUE, 
#     sortVars=TRUE, 
#     labels=names(trainingData),
#     cex.axis=.7, 
#     gap=3, 
#     ylab=c("Histogram of NA values","Pattern"), 
#     digits=2)

####Remove columns with all NA values

trainingData_noNA <- trainingData[,colSums(is.na(trainingData))<nrow(trainingData)]

#-Remove rows with all NA values

trainingData_noNA <- trainingData_noNA[rowSums(is.na(trainingData_noNA[,1:465])) 
                                       != ncol(trainingData_noNA[,1:465]),]
#convert NA's to -110
trainingData_noNA[is.na(trainingData_noNA)] <- -110

##### Matching all colnames of training & validation data
colnames_training <- names(trainingData_noNA)
colnames_training
validationData_noNA <- dplyr::select(validationData, colnames_training)

#convert NA's to -110
validationData_noNA[is.na(validationData_noNA)] <- -110
validationData_noNA

####Explore the Data
# image of reference point locations in data set

plot_ly(x = trainingData_noNA$LONGITUDE, y=trainingData_noNA$LATITUDE, z=trainingData_noNA$FLOOR, 
        type="scatter3d", mode="markers", color = trainingData_noNA$FLOOR) %>%
  add_markers() %>% layout(scene = list(xaxis = list(title = 'Longitude'),
                                        yaxis = list(title = 'Latitude'),
                                        zaxis = list(title = 'Floor')))




#Wap distribution
building0 = c()
building1 = c()
building2 = c()

df1 <- data.frame(building0,building1,building2)
df

data <- c(building0,building1,building2)
df <- data.frame(data)
attributes (df)


for (val in c(1:250)) {
  if (length(which(!is.na(wifi_trainData[wifi_trainData$BUILDINGID == 0,val]))) != 0) 
    building0 = c(building0, val)
  
}



for (val in c(1:250)) {
  if (length(which(!is.na(wifi_trainData[wifi_trainData$BUILDINGID == 1,val]))) != 0) 
    building1 = c(building1, val)
  
}


for (val in c(1:250)) {
  if (length(which(!is.na(wifi_trainData[wifi_trainData$BUILDINGID == 2,val]))) != 0) 
    building2 = c(building2, val)
  
}


boxplot(building0,building1,building2,
        ylab="Distribution of Detected Wireless Access Points by Building")



ggplot(df , aes(x=df, y=)) +
  geom_boxplot(fill='lightblue') +
  theme(text = element_text(size=14)) +
  ggtitle('Distribution of Detected Wireless Access Points by Building') +
  labs(x="Building Number", y= 'WAP Counts' ) +
  theme(panel.border=element_rect(colour='black', fill=NA))


#-Plot histogram of instance counts at locations
frequencyID <- trainingData %>%
  group_indices(BUILDINGID, FLOOR, SPACEID)

fID <- as.data.frame(table(frequencyID))

ggplot(fID, aes(x = Freq)) +
  geom_histogram(fill='brown', binwidth = 2, color='black')+
  scale_x_continuous(breaks=seq(0,100,10)) +
  ggtitle('Frequency Count of Location ID Instances') +
  xlab('Number of Instances for a Loacation ID') +
  ylab('Frequency of Observed Instance Count') +
  theme(text = element_text(size=14)) +
  theme(panel.border=element_rect(colour='black', fill=NA))

##Distribution of WAP count 
plot ( x = as.factor(rep(colnames(trainingData)[1:520],nrow(trainingData))), 
       y = c(t(trainingData[,1:520])))
plot ( x = as.factor(rep(colnames(trainingData_noNA)[1:465],nrow(trainingData_noNA))), 
       y = c(t(trainingData_noNA[,1:465])))

#### Build Predictive Models

#Assign values in seeds agrument of trainControl
set.seed(123)
seeds <- vector(mode = 'list', length=31)
for(i in 1:30) seeds[[i]] <- sample.int(1000, 518)

#for last model:
seeds[[31]] <- sample.int(1000,1)

#-Define parameters in trainControl
ctrl <- trainControl(method = "repeatedcv", 
                     number = 5,
                     repeats=3,
                     allowParallel = TRUE
)

####KNNN MODEL

#Building ID( Removing the attributes  that won’t be needed in a predictive model. 

ttraininingData_noNA <- dplyr::select(trainingData_noNA, -RELATIVEPOSITION, -USERID, 
                               -PHONEID, -TIMESTAMP, -LONGITUDE,
                               -LATITUDE,  -SPACEID, -FLOOR)


tvalidationdata_noNA <- dplyr::select(validationData_noNA, -RELATIVEPOSITION, -USERID, 
                                 -PHONEID, -TIMESTAMP, -LONGITUDE,
                                 -LATITUDE,  -SPACEID, -FLOOR)


set.seed(123)
data_partition <- createDataPartition(y=ttraininingData_noNA$BUILDINGID, p = .7, list = FALSE)
train = ttraininingData_noNA[sample(nrow(data_partition)),]
testing <- ttraininingData_noNA[-data_partition,]


#fit KNN model
set.seed(1234)

#-Grid of k values to search 
knn_grid <- expand.grid(.k=c(1:5))

knn_fit <- train(BUILDINGID ~., data = train,
                 method='knn',
                 preProcess = c('zv'),
                 tuneGrid=knn_grid,
                 trControl = ctrl)
knn_fit

saveRDS(knn_fit, 'knn_fit.rds') 
getwd()

knn_fit_from_file <- readRDS('knn_fit.rds')
plot(knn_fit)
knn_fit$results


#testset

knn_Pred <- predict(knn_fit, newdata = testing)
knn_Pred
knn_CM <-confusionMatrix(knn_Pred, testing$BUILDINGID)
knn_CM
postResample(knn_Pred, testing$BUILDINGID)


# Validationset

knn_Predvd <- predict(knn_fit, newdata = tvalidationdata_noNA)
knn_Predvd
knn_CM <-confusionMatrix(knn_Pred, testing$BUILDINGID)
knn_CM
postResample(knn_Pred, testing$BUILDINGID)


###Floor

ttraininingData_noNAFL <- dplyr::select(trainingData_noNA, -RELATIVEPOSITION, -USERID, 
                               -PHONEID, -TIMESTAMP, -LONGITUDE,
                               -LATITUDE,  -SPACEID, -BUILDINGID)


tvalidationdata_noNAFL <- dplyr::select(validationData_noNA, -RELATIVEPOSITION, -USERID, 
                               -PHONEID, -TIMESTAMP, -LONGITUDE,
                               -LATITUDE,  -SPACEID, -BUILDINGID)


set.seed(123)
data_partition <- createDataPartition(y=ttraininingData_noNAFL$FLOOR, p = .7, list = FALSE)
train = ttraininingData_noNAFL[sample(nrow(data_partition)),]
testing <- ttraininingData_noNAFL[-data_partition,]


knn_fitfloor <- train(FLOOR ~., data = train,
                      method='knn',
                      preProcess = c('zv'),
                      tuneGrid=knn_grid,
                      trControl = ctrl)
knn_fitfloor

plot(knn_fitfloor)

#test set

knn_Predfloor <- predict(knn_fitfloor, newdata = testing)
knn_Predfloor
knn_CMF <-confusionMatrix(knn_Predfloor, testing$FLOOR)
knn_CMF
postResample(knn_Predfloor, testing$FLOOR)

#validation data

knn_Predfloorvd <- predict(knn_fitfloor, newdata = tvalidationdata_noNAFL)
knn_Predfloorvd
postResample(knn_Predfloorvd, tvalidationdata_noNAFL$FLOOR)

### Latitude

ttraininingData_noNAlat <- dplyr::select(trainingData_noNA, -RELATIVEPOSITION, -USERID, 
                                 -PHONEID, -TIMESTAMP, -LONGITUDE,
                                 -FLOOR,  -SPACEID, -BUILDINGID)


tvalidationdata_noNAlat <- dplyr::select(validationData_noNA, -RELATIVEPOSITION, -USERID, 
                                 -PHONEID, -TIMESTAMP, -LONGITUDE,
                                 -FLOOR,  -SPACEID, -BUILDINGID)

set.seed(123)
data_partition <- createDataPartition(y=ttraininingData_noNAlat$LATITUDE, p = .7, list = FALSE)
trainlat = ttraininingData_noNAlat[sample(nrow(data_partition)),]
testinglat <- ttraininingData_noNAlat[-data_partition,]


knn_fitlatitude <- train(LATITUDE ~., data = trainlat,
                         method='knn',
                         preProcess = c('zv'),
                         tuneGrid=knn_grid,
                         trControl = ctrl)

knn_fitlatitude
plot(knn_fitlatitude)

#test set
knn_Predlatitude <- predict(knn_fitlatitude, newdata = testinglat)
knn_Predlatitude
postResample(knn_Predlatitude, testinglat$LATITUDE)

#validation data
knn_Predlatvd <- predict(knn_fitlatitude, newdata = tvalidationdata_noNAlat)
knn_Predlatvd
plot(knn_Predlatvd)
postResample(knn_Predlatvd, tvalidationdata_noNAlat$LATITUDE)

###Longitude
ttraininingData_noNAlon <- dplyr ::select(trainingData_noNA, -RELATIVEPOSITION, -USERID, 
                                  -PHONEID, -TIMESTAMP, -LATITUDE,
                                  -FLOOR,  -SPACEID, -BUILDINGID)
tvalidationdata_noNAlon <- dplyr::select(validationData_noNA, -RELATIVEPOSITION, -USERID, 
                                  -PHONEID, -TIMESTAMP, -LATITUDE,
                                  -FLOOR,  -SPACEID, -BUILDINGID)


set.seed(123)
data_partition <- createDataPartition(y=ttraininingData_noNAlon$LONGITUDE, p = .7, list = FALSE)
train = ttraininingData_noNAlon[sample(nrow(data_partition)),]
testing <- ttraininingData_noNAlon[-data_partition,]

knn_fitlongitude <- train(LONGITUDE ~., data = train,
                         method='knn',
                         preProcess = c('zv'),
                         tuneGrid=knn_grid,
                         trControl = ctrl)
knn_fitlongitude
plot(knn_fitlongitude)

#test data set
knn_Predlongi <- predict(knn_fitlongitude, newdata = testing)
knn_Predlongi

postResample(knn_Predlongi, testing$LONGITUDE)

#validation data
knn_Predlonvd <- predict(knn_fitlongitude, newdata = tvalidationdata_noNAlon)
knn_Predlonvd

postResample(knn_Predlonvd, tvalidationdata_noNAlon$LONGITUDE)


##### decision tree (C5.0) model

###Building ID

set.seed(123)

d_fit <- train(BUILDINGID~., data = train,
                          method='C5.0',
                          preProcess = c('zv'),
                          trControl = ctrl)

d_fit
plot(d_fit)

#testset

dfit_Pred <- predict(d_fit, newdata = testing)
dfit_Pred
dfit_CM <-confusionMatrix(dfit_Pred, testing$BUILDINGID)
dfit_CM
postResample(dfit_Pred, testing$BUILDINGID)


## Validationset

dfit_Predvd <- predict(d_fit, newdata = tvalidationdata_noNA)
dfit_Predvd
dfit_CMvd <-confusionMatrix(dfit_Predvd, tvalidationdata_noNA$BUILDINGID)
dfit_CMvd
postResample(dfit_Predvd, tvalidationdata_noNA$BUILDINGID)

### Floor
d_fitfloor <- train(FLOOR~., data = train,
               method='C5.0',
               preProcess = c('zv'),
               trControl = ctrl)
d_fitfloor
plot(d_fitfloor)

d_Predfloor <- predict(d_fitfloor, newdata = testing)
d_Predfloor
d_CMF <-confusionMatrix(d_Predfloor, testing$FLOOR)
d_CMF
postResample(d_Predfloor, testing$FLOOR)

#validation data

d_Predfloorvd <- predict(d_fitfloor, newdata = tvalidationdata_noNAFL)
d_Predfloorvd
postResample(d_Predfloorvd, tvalidationdata_noNAFL$FLOOR)


####Random Forest

##Building

rf_fit <- train(BUILDINGID ~., data = train,
                 method ='ranger',
                 preProcess = c('zv'),
                 trControl = ctrl)
rf_fit
plot(rf_fit)

#testset
rf_Pred <- predict(rf_fit, newdata = testing)
rf_Pred
rf_CM <-confusionMatrix(rf_Pred, testing$BUILDINGID)
rf_CM
postResample(rf_Pred, testing$BUILDINGID)


## Validationset

rf_Predvd <- predict(rf_fit, newdata = tvalidationdata_noNA)
rf_Pred
rf_CMvd <-confusionMatrix(rf_Pred, testing$BUILDINGID)
rf_CMvd
postResample(rf_Pred, testing$BUILDINGID)


##FLOOR
set.seed(123)

rf_fitfloor <- train(FLOOR ~., data = train,
                            method = 'ranger',
                         preProcess = c('zv'),
                         trControl = ctrl)

rf_fitfloor
plot(rf_fitfloor)

#test set
rf_Predfloor <- predict(rf_fitfloor, newdata = testing)
rf_Predfloor
rf_CMF <-confusionMatrix(rf_Predfloor, testing$FLOOR)
rf_CMF
postResample(rf_Predfloor, testing$FLOOR)

#validation data
rf_Predfloorvd <- predict(rf_fitfloor, newdata = tvalidationdata_noNAFL)
rf_Predfloorvd
postResample(rf_Predfloorvd, tvalidationdata_noNAFL$FLOOR)


#latitude
rf_fitlatitude <- train(LATITUDE ~., data = trainlat,
                         method='ranger',
                         preProcess = c('zv'),
                         trControl = ctrl)

rf_fitlatitude

plot(rf_fitlatitude)

#test set

rf_Predlatitude <- predict(rf_fitlatitude, newdata = testinglat)
rf_Predlatitude
postResample(rf_Predlatitude, testinglat$LATITUDE)

#validation data
rf_Predlatvd <- predict(rf_fitlatitude, newdata = tvalidationdata_noNAlat)
rf_Predlatvd
postResample(rf_Predlatvd, tvalidationdata_noNAlat$LATITUDE)


### Longitiude

rf_fitlongitude <- train(LONGITUDE ~., data = train,
                          method='ranger',
                          preProcess = c('zv'),
                          trControl = ctrl)
rf_fitlongitude
plot(rf_fitlongitude)


#test set
rf_Predlongi <- predict(rf_fitlongitude, newdata = testing)
rf_Predlongi
postResample(rf_Predlongi, testing$LONGITUDE)

#validation data
rf_Predlonvd <- predict(rf_fitlongitude, newdata = tvalidationdata_noNAlon)
rf_Predlonvd
postResample(rf_Predlonvd, tvalidationdata_noNAlon$LONGITUDE)




#SVM Model
set.seed(123)
svm_fit <- train(BUILDINGID ~., data = train,
                method='svmLinear',
                preProcess = c('zv'),
                trControl = ctrl)
svm_fit
plot(svm_fit)

#testset

svm_Pred <- predict(svm_fit, newdata = testing)
svm_Pred
plot(svm_Pred)
svm_CM <-confusionMatrix(svm_Pred, testing$BUILDINGID)
svm_CM
postResample(svm_Pred, testing$BUILDINGID)

# validation set
svm_Predvd <- predict(svm_fit, newdata = tvalidationdata_noNA)
svm_Pred
svm_CM <-confusionMatrix(svm_Pred, testing$BUILDINGID)
svm_CM
postResample(svm_Pred, testing$BUILDINGID)

#Floor

svm_fitfloor <- train(FLOOR ~., data = train,
                      method='svmLinear',
                      preProcess = c('zv'),
                      trControl = ctrl)
svm_fitfloor
plot(svm_fitfloor)

#test set
svm_Predfloor <- predict(svm_fitfloor, newdata = testing)
svm_Predfloor
svm_CMF <-confusionMatrix(svm_Predfloor, testing$FLOOR)
svm_CMF
postResample(svm_Predfloor, testing$FLOOR)

#validation data

svm_Predfloorvd <- predict(svm_fitfloor, newdata = tvalidationdata_noNAFL)
svm_Predfloorvd
postResample(svm_Predfloorvd, tvalidationdata_noNAFL$FLOOR)


#latitude
set.seed (123)
svm_fitlatitude <- train(LATITUDE ~., data = trainlat,
                         method='svmLinear',
                         preProcess = c('zv'),
                         trControl = ctrl)

svm_fitlatitude

# test data
svm_Predlatitude <- predict(svm_fitlatitude, newdata = testinglat)
svm_Predlatitude
postResample(svm_Predlatitude, testinglat$LATITUDE)

#validation data
svm_Predlatvd <- predict(svm_fitlatitude, newdata = tvalidationdata_noNAlat)
svm_Predlatvd
postResample(svm_Predlatvd, tvalidationdata_noNAlat$LATITUDE)

# Longitude

svm_fitlongitude <- train(LONGITUDE ~., data = train,
                          method='svmLinear',
                          preProcess = c('zv'),
                          trControl = ctrl)
svm_fitlongitude


#test data
svm_Predlongi <- predict(svm_fitlongitude, newdata = testing)
svm_Predlongi

postResample(svm_Predlongi, testing$LONGITUDE)

#validation data
svm_Predlonvd <- predict(svm_fitlongitude, newdata = tvalidationdata_noNAlon)
svm_Predlonvd

postResample(svm_Predlonvd, tvalidationdata_noNAlon$LONGITUDE)


####Comparison of Models

#summary results classificateion
#buildings
results <- resamples(list(kNN=knn_fit,
                          RF=rf_fit,
                          svm= svm_fit,
                          C5.0=d_fit))
summary(results)
plot(results)
bwplot(results)

# Floor 
resultsfloor <- resamples(list(kNN=knn_fitfloor,
                               c5.0 = d_fitfloor,
                               svm = svm_fit,
                               RF = rf_fitfloor))

summary(resultsfloor)
bwplot(resultsfloor)


#latitude
resultslatitude <- resamples(list(knn=knn_fitlatitude,
                                  RF = rf_fitlatitude,
                                  svm = svm_fitlatitude))
summary(resultslatitude)
bwplot(resultslatitude)

#longitude
resultslongitude <- resamples(list(knn= knn_fitlongitude,
                                   RF = rf_fitlongitude,
                                   svm = svm_fitlongitude))
summary(resultslongitude)
bwplot(resultslongitude)

#Distribution of distance error (in meters)
Error = sqrt((knn_Predlonvd - validationData_noNA$LONGITUDE)^2
             +(knn_Predlatvd - validationData_noNA$LATITUDE)^2)

mean(Error)

hypotenuse(6.12,5.75)

hist(Error, freq = T, xlab = " Absolute error (m)", 
     main = "Error distance in meters (Buildings)")
summary ('Error')


# Plots latitude & Longitude

scatterplot3d(knn_Predlonvd, knn_Predlatvd, knn_Predfloorvd,
              type='p',
              highlight.3d = FALSE,
              color='blue',
              angle=120,
              pch=16,
              box=FALSE,
              main = "Location Reference Points",
              cex.lab = 1,
              cex.main=1,
              cex.sub=1,
              col.sub='red',
              xlab='Longitude', ylab='Latitude',zlab = 'Building Floor') 



scatterplot3d(validationData_noNA$LONGITUDE, validationData_noNA$LATITUDE, validationData_noNA$FLOOR,
              type='p',
              highlight.3d = FALSE,
              color='brown',
              angle=155,
              pch=16,
              box=FALSE,
              main = "Location Reference Points",
              cex.lab = 1,
              cex.main=1,
              cex.sub=1,
              col.sub='red',
              xlab='Longitude', ylab='Latitude',zlab = 'Building Floor') 


 plot_ly(x = validationData_noNA$LONGITUDE, y=validationData_noNA$LATITUDE, z=validationData_noNA$FLOOR, 
                 type="scatter3d", mode="markers", color = validationData_noNA$FLOOR) %>%
  add_markers() %>% layout(scene = list(xaxis = list(title = 'Longitude'),
                                        yaxis = list(title = 'Latitude'),
                                        zaxis = list(title = 'Floor')))



 plot_ly(x = knn_Predlonvd, y=knn_Predlatvd, z=knn_Predfloorvd, 
                type="scatter3d", mode="markers", color = knn_Predfloorvd) %>%
  add_markers() %>% layout(scene = list(xaxis = list(title = 'Longitude'),
                                        yaxis = list(title = 'Latitude'),
                                        zaxis = list(title = 'Floor')))
 
 




colors <- c( rep("#E69F00", 10972), rep("#56B4E9",10000))


df2$category [df$category == "training"]  = 0
df2$category [df$category == "predict"]  = 1


scatterplot3d(df2$lon, df2$lat, df2$floor,
              type='p',
              highlight.3d = FALSE,
              color=  "#56B4E9", "#E69F00",
              angle=120,
              pch=16,
              box=FALSE,
              main = "Location Reference Points",
              cex.lab = 1,
              cex.main=1,
              cex.sub=1,
              col.sub='red',
              xlab='Longitude', ylab='Latitude',zlab = 'Building Floor') 

table(df2$category)

scatterplot3d(df2$lon, df2$lat, df2$floor,
              type='p',
              highlight.3d = FALSE,
              color=  colors,
              angle=120,
              pch=16,
              box=FALSE,
              main = "Location Reference Points",
              cex.lab = 1,
              cex.main=1,
              cex.sub=1,
              col.sub='red',
              xlab='Longitude', ylab='Latitude',zlab = 'Building Floor') 









