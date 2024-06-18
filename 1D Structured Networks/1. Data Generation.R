## Load in the necessary libraries
library(dplyr)
library(tidyr)
library(keras)
library(tidymodels)
library(caret)
library(tensorflow)

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
set.seed(73)
split <- group_initial_split(processed_data, group = fishNum, strata = species, prop = 0.7)
train <- training(split)
val_test <- testing(split)
split2 <- group_initial_split(val_test, group = fishNum, strata = species, prop = 0.5)
validate <- training(split2)
test <- testing(split2)

# Check the dataset set
train%>%group_by(species)%>%count()
validate%>%group_by(species)%>%count()
test%>%group_by(species)%>%count()

# Dummy encode y data
train$y <- ifelse(train$species == "LT", 0, 1)
dummy_y_train <- to_categorical(train$y, num_classes = 2)
test$y <- ifelse(test$species == "LT", 0, 1)
dummy_y_test <- to_categorical(test$y, num_classes = 2)
validate$y <- ifelse(validate$species == "LT", 0, 1)
dummy_y_val <- to_categorical(validate$y, num_classes = 2)

# Transform and standardize training data 
x_train <- train %>% select(52:300)
x_train<-x_train+10*log10(450/train$totalLength)
x_train<-exp(x_train/10)
x_train<-x_train%>%scale()
x_train<-as.matrix(x_train)

# Save the mean and sd to apply to the validation and test set
xmean<-attributes(x_train)$`scaled:center`
xsd<-attributes(x_train)$`scaled:scale`

# Transform and standardize validation data 
x_validate <- validate %>% select(52:300)
x_validate<-x_validate+10*log10(450/validate$totalLength)
x_validate<-exp(x_validate/10)
x_validate<-x_validate%>%scale(xmean,xsd)
x_validate<-as.matrix(x_validate)

# Transform and standardize training data 
x_test <- test %>% select(52:300)
x_test<-x_test+10*log10(450/test$totalLength)
x_test<-exp(x_test/10)
x_test<-x_test%>%scale(xmean,xsd)
x_test<-as.matrix(x_test)

# Shuffle training data
set.seed(250)
train_indices <- sample(1:nrow(x_train))
x_train <- x_train[train_indices, ] 
dummy_y_train <- dummy_y_train[train_indices, ] 

# Shuffle validation data
set.seed(250)
val_indices <- sample(1:nrow(x_validate))
x_validate <- x_validate[val_indices, ] 
dummy_y_val <- dummy_y_val[val_indices, ]
