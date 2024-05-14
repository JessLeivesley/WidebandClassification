# Data generation for DNN, 1D CNN, and ResNet

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(groupdata2)
library(keras)

## ---- Setting up data ---- 

# Load in data
load("Data/processed_AnalysisData_no200.Rdata")

#rename to make typing it easier
processed_data<-processed_data_no200

# Remove individuals with missing transducers
processed_data<-processed_data%>%filter(is.na(F100)==F)

# Remove the two missing frequency columns
processed_data<-processed_data%>%select(-F90)
processed_data<-processed_data%>%select(-F90.5)

# Include only Lake Trout and Smallmouth Bass individuals
processed_data<-processed_data%>%filter(spCode == "81" |spCode == "316")

# Create species label column
processed_data$species<-ifelse(processed_data$spCode==81, "LT","SMB")

# Remove missing values
processed_data<-processed_data%>%filter(is.na(aspectAngle)==F & is.na(Angle_major_axis)==F)

# Remove one outlier
processed_data<-processed_data%>%filter(F100>-1000)

## ---- Create train/test split ----
set.seed(73)
split<-group_initial_split(processed_data,group=fishNum,strata = species, prop=0.85)
train<-training(split)
test<-testing(split)

# Check how balanced the datasets are (roughly 2.1-2.2 times more LT than SMB)
train%>%group_by(species)%>%count() 
test%>%group_by(species)%>%count() 

## ---- Create cross validation folds ----

# Transform fishNum (unique fish ID) from a character to a factor
train$fishNum<-as.factor(train$fishNum)

set.seed(15)
train.cv <- groupdata2::fold(train, k = 5, cat_col = 'species', id_col = 'fishNum')
train.cv$`.folds`

## ---- Dummy encode y variable ----
train$y<-NA
train$y[train$species=="LT"]<-0
train$y[train$species=="SMB"]<-1
summary(train$y)
dummy_y_train<-to_categorical(train$y, num_classes = 2)


test$y<-NA
test$y[test$species=="LT"]<-0
test$y[test$species=="SMB"]<-1
summary(test$y)
dummy_y_test<-to_categorical(test$y, num_classes = 2)

## ---- Create x data matrices and normalize data----

# select the positional data (cols 21,23,24) and the TS data (cols 52:300)
x_train <- train %>% 
  select(c(21,23,24,52:300))

# Adjust TS values to a set fish length
x_train[,4:252]<-x_train[,4:252]+10*log10(450/train$totalLength)

# Transform from TS to acoustic backscatter
x_train[,4:252]<-exp(x_train[,4:252]/10)
x_train<-as.matrix(x_train)

# Do the same for the test data
x_test <- test %>% 
  select(c(21,23,24,52:300))
x_test[,4:252]<-x_test[,4:252]+10*log10(450/test$totalLength)
x_test[,4:252]<-exp(x_test[,4:252]/10)
x_test<-as.matrix(x_test)

# add folds to x matrix and y data
x_train<-cbind(x_train,train.cv$`.folds`)
dummy_y_train<-cbind(dummy_y_train,train.cv$.folds)

## Shuffle training data
set.seed(250)
x<-sample(1:nrow(x_train))
x_train_S= x_train[x, ] 
dummy_y_train_S= dummy_y_train[x, ] 