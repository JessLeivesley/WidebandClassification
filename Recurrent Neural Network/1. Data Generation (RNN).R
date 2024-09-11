# Pre-process the data for recurrent neural network cross validation

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(groupdata2)
library(keras)
library(str2str)
library(caret)

## ---- Setting up data ---- 
# Load in the dataset
load("Data/processed_AnalysisData_no200.Rdata")

processed_data <- processed_data_no200

processed_data%>%group_by(spCode,fishNum)%>%count()

processed_data <- processed_data %>% filter(is.na(F100) == F)
processed_data <- processed_data %>% filter(fishNum != "LWF23018")
processed_data <- processed_data %>% select(-F90, -F90.5)
processed_data <- processed_data %>% filter(spCode == "81" | spCode == "316")
processed_data$species <- ifelse(processed_data$spCode == 81, "LT", "SMB")
processed_data <- processed_data %>% filter(is.na(aspectAngle) == F & is.na(Angle_major_axis) == F & is.na(Angle_minor_axis) == F)
processed_data <- processed_data %>% filter(F100 > -1000)

# Split into training/validation/test set
# Might need to change up the splitting proportions
set.seed(73)
split <- group_initial_split(processed_data, group = fishNum, strata = species, prop = 0.7)
train <- training(split)
val_test <- testing(split)
split2 <- group_initial_split(val_test, group = fishNum, strata = species, prop = 0.5)
validate <- training(split2)
test <- testing(split2)


# transform and standardize data
train_st <- train %>% 
  select(52:300)
train_st<-train_st+10*log10(450/train$totalLength)
train_st<-exp(train_st/10)
train_st<-as.data.frame(train_st%>%scale())

# adding back necessary/important columns
train_st$species<-train$species
train_st$Region<-train$Region_name
train_st$fishNum<-train$fishNum

validate_st <- validate %>% 
  select(52:300)
validate_st<-validate_st+10*log10(450/validate$totalLength)
validate_st<-exp(validate_st/10)
validate_st<-as.data.frame(validate_st%>%scale())

# adding back necessary/important columns
validate_st$species<-validate$species
validate_st$Region<-validate$Region_name
validate_st$fishNum<-validate$fishNum

test_st <- test %>% 
  select(52:300)
test_st<-test_st+10*log10(450/test$totalLength)
test_st<-exp(test_st/10)
test_st<-as.data.frame(test_st%>%scale())

# adding back necessary/important columns
test_st$species<-test$species
test_st$Region<-test$Region_name
test_st$fishNum<-test$fishNum



## ---- Create groups of 5 pings for the timeseries ----

# Doing this separately for teach of the splits 
# Training set first

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
train_grps<-train_st%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
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

# Validation data
# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
val_grps<-validate_st%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
head(val_grps)

# splitting into lists 
listgrps_val<-val_grps%>%group_split(Region,grp)

# keeping only lists that are of length 5
listgrps_val<-listgrps_val[sapply(listgrps_val, nrow) >= 5]

# select frequencies only
listgrps_val2<-map(listgrps_val, ~ (.x %>% select(c(1:249))))

# each dataframe in the list to a matrix
x_data_val<-lapply(listgrps_val2, as.matrix)

# Flatten into a 3D array
x_data_val<-lm2a(x_data_val,dim.order=c(3,1,2))

# Check dims
dim(x_data_val)

# Selecting the y data
y_data_val<-vector()

for(i in 1:dim(x_data_val)[1]){
  a <-listgrps_val[[i]]%>%select(species)
  y_data_val[i]<-a[1,]
}

# Unlist
y_data_val<-unlist(y_data_val)

# Dummy code this
y_val<-NA
y_val[y_data_val=="LT"]<-0
y_val[y_data_val=="SMB"]<-1
summary(y_val)
dummy_y_val<-to_categorical(y_val, num_classes = 2)
dim(dummy_y_val)

# Now for testing data

# Creating a listing variable within each group so that we can split groups longer than 5 into groups of 5
test_grps<-test_st%>%group_by(Region)%>%mutate(grp=rep(1:ceiling(n()/5), each=5, length.out=n()))%>%ungroup()
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

set.seed(15)
shuffle_index_val<-sample(1:dim(x_data_val)[1],dim(x_data_val)[1])
x_data_val<-x_data_val[shuffle_index_val,,]
dummy_y_val<-dummy_y_val[shuffle_index_val,]
