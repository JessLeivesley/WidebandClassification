# Pre-process the data for recurrent neural network cross validation

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(groupdata2)
library(keras)
library(str2str)
library(caret)

## ---- Setting up data ---- 

# Load in data
load("Data/processed_AnalysisData_no200.Rdata")

# make the name easier to type
processed_data<-processed_data_no200

# Look at the structure of individuals
processed_data%>%group_by(spCode,fishNum,Region_name)%>%dplyr::count()

# Create unique region per fish
processed_data$Region<-interaction(processed_data$fishNum,processed_data$Region_name)

# Remove individuals with missing transducers
processed_data<-processed_data%>%filter(is.na(F100)==F)

# also remove individual LWF23018 (only two pings)
processed_data<-processed_data%>%filter(fishNum!="LWF23018")

# remove the 90kHZ and 90.5kHZ columns
processed_data<-processed_data%>%select(-F90)
processed_data<-processed_data%>%select(-F90.5)

# only keep LT (81), and SMB (316)
processed_data<-processed_data%>%filter(spCode == "81" |spCode == "316")

processed_data$species<-ifelse(processed_data$spCode==81, "LT", 
                               ifelse(processed_data$spCode == 91, "LWF", "SMB"))

# remove the one ping that has a VERY low TS
processed_data<-processed_data%>%filter(F100>-1000)
glimpse(processed_data)

# transform and standardize data
processed_data_TS <- processed_data %>% 
  select(52:300)
processed_data_TS<-processed_data_TS+10*log10(450/processed_data$totalLength)
processed_data_TS<-exp(processed_data_TS/10)
processed_data_TS<-as.data.frame(processed_data_TS%>%scale())

# adding back necessary/important columns
processed_data_TS$species<-processed_data$species
processed_data_TS$Region<-processed_data$Region
processed_data_TS$fishNum<-processed_data$fishNum

## ---- Create train/test split ----

# group by fishNum makes sure same fish doesn't appear in both testing and training
set.seed(15)
split<-group_initial_split(processed_data_TS,group=fishNum,strata = species, prop=0.9)
train<-training(split)
test<-testing(split)

## ---- Create groups of 5 pings for the timeseries ----

# Doing this separately for the test and training set
# Training set first

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
train_grps<-train%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(train_grps)

# splitting into lists 
listgrps_train<-train_grps%>%group_split(Region,grp)

# keeping only lists that are of length 5
listgrps_train<-listgrps_train[sapply(listgrps_train, nrow) >= 5]

# select frequencies only
listgrps_train2<-map(listgrps_train, ~ (.x %>% select(c(1:249))))

# each dataframe in the list to a matrix
x_data_train<-lapply(listgrps_train2, as.matrix)

# Flatten into a 3D array
x_data_train<-lm2a(x_data_train,dim.order=c(3,1,2))

# Check dims
dim(x_data_train)

# Selecting the y data
y_data_train<-vector()

for(i in 1:dim(x_data_train)[1]){
  a <-listgrps_train[[i]]%>%select(species)
  y_data_train[i]<-a[1,]
}

# Unlist
y_data_train<-unlist(y_data_train)

# Dummy code this
y_train<-NA
y_train[y_data_train=="LT"]<-0
y_train[y_data_train=="SMB"]<-1
summary(y_train)
dummy_y_train<-to_categorical(y_train, num_classes = 2)
dim(dummy_y_train)

## Getting fish ID so that it is not repeated across folds
fishID_train<-vector()

for(i in 1:dim(x_data_train)[1]){
  a <-listgrps_train[[i]]%>%select(fishNum)
  fishID_train[i]<-a[1,]
}

# Unlist
fishID_train<-unlist(fishID_train)

# Now for testing data

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
test_grps<-test%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(test_grps)

# splitting into lists 
listgrps_test<-test_grps%>%group_split(Region,grp)

# keeping only lists that are of length 5
listgrps_test<-listgrps_test[sapply(listgrps_test, nrow) >= 5]

# keeping only frequencies
listgrps_test2<-map(listgrps_test, ~ (.x %>% select(c(1:249))))

# each data frame in the list to a matrix
x_data_test<-lapply(listgrps_test2, as.matrix)

# Flatten into a 3D array
x_data_test<-lm2a(x_data_test,dim.order=c(3,1,2))

# Check dims
dim(x_data_test)

# Selecting the y data
y_data_test<-vector()

for(i in 1:dim(x_data_test)[1]){
  a <-listgrps_test[[i]]%>%select(species)
  y_data_test[i]<-a[1,]
}

# Unlist
y_data_test<-unlist(y_data_test)

# Dummy code this
y_test<-NA
y_test[y_data_test=="LT"]<-0
y_test[y_data_test=="SMB"]<-1
summary(y_test)
dummy_y_test<-to_categorical(y_test, num_classes = 2)
dim(dummy_y_test)

## shuffle data 
set.seed(15)
shuffle_index_train<-sample(1:dim(x_data_train)[1],dim(x_data_train)[1])
x_data_train<-x_data_train[shuffle_index_train,,]
dummy_y_train<-dummy_y_train[shuffle_index_train,]
fishID_train<-fishID_train[shuffle_index_train]

## ---- Create 5 groups for cross validation ----
# K fold validation, but with no repeat of fish across, preventing data leakage
set.seed(15)
folds<-groupKFold(fishID_train,k=5)
