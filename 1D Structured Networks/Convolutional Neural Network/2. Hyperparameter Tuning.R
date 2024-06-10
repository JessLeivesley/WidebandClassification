# Hyperparameter Tuning for ResNet

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(caret)
library(rsample)
library(keras)

## ---- Data set up ----
# Remove the positional variables from the data sets generated in 1. Data Generation

x_train <- as.data.frame(x_train_S)%>%select(-aspectAngle, -Angle_minor_axis,-Angle_major_axis)
x_test <- as.data.frame(x_test)%>%select(-aspectAngle, -Angle_minor_axis,-Angle_major_axis)

x_train<-as.matrix(x_train)
x_test<-as.matrix(x_test)

## ---- Grid Search ----
# create grid of parameter space we want to search
filters <- c(2, 4, 8, 16)
kernel_size <- c(3, 5, 7)
batch_size <- c(100, 300, 500, 800, 1000, 1200, 1500)
multipliers1 <-c(1,2)
multipliers2 <-c(1,2)
multipliers3 <-c(1,2)
multipliers4 <-c(1,2)
droprate1=c(0,0.1,0.15)
droprate2=c(0,0.1,0.15)
droprate3=c(0,0.1,0.15)
droprate4=c(0,0.1,0.15)
droprate5=c(0,0.1,0.15)
regularizer_weight <- c(0, 0.001,0.005,0.01)

# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full<-expand.grid(filters = filters, kernel_size = kernel_size, multipliers1 = multipliers1,
                              multipliers2 = multipliers2, multipliers3 = multipliers3, multipliers4 = multipliers4,
                              batch_size = batch_size, droprate=droprate1,droprate2=droprate2,
                              droprate3=droprate3, droprate4=droprate4, droprate5=droprate5,
                              regularizer_weight = regularizer_weight)

set.seed(15)
# x<-sample(1:45,20,replace=F) # 45 = size of grid search (num of row)
n_subset <- 20
x<-sample(1:nrow(grid.search.full), n_subset,replace=F)
grid.search.subset<-grid.search.full[x,]

val_loss<-matrix(nrow=n_subset,ncol=5)
best_epoch_loss<-matrix(nrow=n_subset,ncol=5)
val_auc<-matrix(nrow=n_subset,ncol=5)
best_epoch_auc<-matrix(nrow=n_subset,ncol=5)

## ---- Run Search ----
# Run in two groups for memory (faster this way)
for (i in 1:10){
  for (fold in 1:5){
    print(grid.search.subset[i,])
    print(sprintf("Processing Fold #%d", fold))

    x_train_set<-x_train[x_train[,250] != fold,]
    y_train_set<-dummy_y_train_S[dummy_y_train_S[,3]!=fold,]

    x_val_set<-x_train[x_train[,250] == fold,]
    y_val_set<-dummy_y_train_S[dummy_y_train_S[,3]==fold,]

    cw<-summary(as.factor(y_train_set[,1]))[2]/summary(as.factor(y_train_set[,1]))[1]

    # Scaling within the loop
    scaler <- preProcess(x_train_set[,1:249], method = 'scale')
    x_train_set <- predict(scaler, x_train_set[,1:249])
    x_val_set <- predict(scaler, x_val_set[,1:249])

    
    # x_train_set<-x_train[fold[[fold]],,]
    # y_train_set<-dummy_y_train[fold[[fold]],]
    # 
    # x_val_set<-x_train[-fold[[fold]],,]
    # y_val_set<-dummy_y_train[-fold[[fold]],]
    set_random_seed(15)
    cnn = keras_model_sequential()
    #model
    cnn %>% 
      layer_conv_1d(input_shape=c(249,1), filters = grid.search.subset$filters[i], kernel_size = grid.search.subset$kernel_size[i], 
                    activation = 'relu', padding = 'same', strides = 1)%>%
      layer_dropout(rate = grid.search.subset$droprate[i])%>%
      layer_batch_normalization()%>%
      layer_conv_1d(filters = grid.search.subset$filters[i] * grid.search.subset$multipliers1[i], kernel_size = grid.search.subset$kernel_size[i],  
                    activation = 'relu', padding = 'same', strides = 1)%>%
      layer_dropout(rate = grid.search.subset$droprate2[i])%>%
      layer_batch_normalization()%>%
      layer_max_pooling_1d(pool_size = 2)%>%
      layer_conv_1d(filters = grid.search.subset$filters[i] * grid.search.subset$multipliers1[i] * grid.search.subset$multipliers2[i], kernel_size = grid.search.subset$kernel_size[i],  
                    activation = 'relu', padding = 'same', strides = 1)%>%
      layer_dropout(rate = grid.search.subset$droprate3[i])%>%
      layer_batch_normalization()%>%
      layer_conv_1d(filters = grid.search.subset$filters[i] * grid.search.subset$multipliers1[i] * grid.search.subset$multipliers2[i] * 
                      grid.search.subset$multipliers3[i], kernel_size = grid.search.subset$kernel_size[i], 
                    activation = 'relu', padding = 'same', strides = 1)%>%
      layer_dropout(rate = grid.search.subset$droprate4[i])%>%
      layer_max_pooling_1d(pool_size = 2)%>%
      layer_batch_normalization()%>%
      layer_conv_1d(filters = grid.search.subset$filters[i] * grid.search.subset$multipliers1[i] * grid.search.subset$multipliers2[i] * 
                      grid.search.subset$multipliers3[i]* grid.search.subset$multipliers4[i], kernel_size = grid.search.subset$kernel_size[i], 
                    activation = 'relu', padding = 'same', strides = 1)%>%
      layer_dropout(rate = grid.search.subset$droprate5[i])%>%
      layer_batch_normalization()%>%
      layer_flatten() %>%
      layer_dense(units = 2, activation="sigmoid")
  
    cnn %>% compile(
      optimizer = optimizer_adam(weight_decay = grid.search.subset$regularizer_weight[i]),
      loss = loss_binary_crossentropy(),
      metrics = c("accuracy", tf$keras$metrics$AUC())
    )
    
    # Fit model (just resnet)
    cnn_history <- cnn %>% fit(
      x_train_set, y_train_set[,c(1:2)],
      batch_size = grid.search.subset$batch_size[i],
      epochs = 50,
      validation_data = list(x_val_set, y_val_set[,c(1:2)]),
      class_weight = list("0"=1,"1"=cw)
    )
    
    val_loss[i,fold]<-min(cnn_history$metrics$val_loss)
    best_epoch_loss[i,fold]<-which(cnn_history$metrics$val_loss==min(cnn_history$metrics$val_loss))
    val_auc[i,fold] <- max(cnn_history$metrics$val_auc)
    best_epoch_auc[i,fold]<-which(cnn_history$metrics$val_auc==max(cnn_history$metrics$val_auc))
    
    print(i)
    print(fold)
    
  }
}

for (i in 11:20){
  for (fold in 1:5){
    print(grid.search.subset[i,])
    print(sprintf("Processing Fold #%d", fold))
    
    x_train_set<-x_train[x_train[,250] != fold,]
    y_train_set<-dummy_y_train_S[dummy_y_train_S[,3]!=fold,]
    
    x_val_set<-x_train[x_train[,250] == fold,]
    y_val_set<-dummy_y_train_S[dummy_y_train_S[,3]==fold,]
    
    cw<-summary(as.factor(y_train_set[,1]))[2]/summary(as.factor(y_train_set[,1]))[1]
    
    # Scaling within the loop
    scaler <- preProcess(x_train_set[,1:249], method = 'scale')
    x_train_set <- predict(scaler, x_train_set[,1:249])
    x_val_set <- predict(scaler, x_val_set[,1:249])
    
    set_random_seed(15)
    cnn = keras_model_sequential()
    #model
    cnn %>% 
      layer_conv_1d(input_shape=c(249,1), filters = grid.search.subset$filters[i], kernel_size = grid.search.subset$kernel_size[i], 
                    activation = 'relu', padding = 'same', strides = 1)%>%
      layer_dropout(rate = grid.search.subset$droprate[i])%>%
      layer_batch_normalization()%>%
      layer_conv_1d(filters = grid.search.subset$filters[i] * grid.search.subset$multipliers1[i], kernel_size = grid.search.subset$kernel_size[i],  
                    activation = 'relu', padding = 'same', strides = 1)%>%
      layer_dropout(rate = grid.search.subset$droprate2[i])%>%
      layer_batch_normalization()%>%
      layer_max_pooling_1d(pool_size = 2)%>%
      layer_conv_1d(filters = grid.search.subset$filters[i] * grid.search.subset$multipliers1[i] * grid.search.subset$multipliers2[i], kernel_size = grid.search.subset$kernel_size[i],  
                    activation = 'relu', padding = 'same', strides = 1)%>%
      layer_dropout(rate = grid.search.subset$droprate3[i])%>%
      layer_batch_normalization()%>%
      layer_conv_1d(filters = grid.search.subset$filters[i] * grid.search.subset$multipliers1[i] * grid.search.subset$multipliers2[i] * 
                      grid.search.subset$multipliers3[i], kernel_size = grid.search.subset$kernel_size[i], 
                    activation = 'relu', padding = 'same', strides = 1)%>%
      layer_dropout(rate = grid.search.subset$droprate4[i])%>%
      layer_max_pooling_1d(pool_size = 2)%>%
      layer_batch_normalization()%>%
      layer_conv_1d(filters = grid.search.subset$filters[i] * grid.search.subset$multipliers1[i] * grid.search.subset$multipliers2[i] * 
                      grid.search.subset$multipliers3[i]* grid.search.subset$multipliers4[i], kernel_size = grid.search.subset$kernel_size[i], 
                    activation = 'relu', padding = 'same', strides = 1)%>%
      layer_dropout(rate = grid.search.subset$droprate5[i])%>%
      layer_batch_normalization()%>%
      layer_flatten() %>%
      layer_dense(units = 2, activation="sigmoid")
    
    cnn %>% compile(
      optimizer = optimizer_adam(weight_decay = grid.search.subset$regularizer_weight[i]),
      loss = loss_binary_crossentropy(),
      metrics = c("accuracy", tf$keras$metrics$AUC())
    )
    
    # Fit model (just resnet)
    cnn_history <- cnn %>% fit(
      x_train_set, y_train_set[,c(1:2)],
      batch_size = grid.search.subset$batch_size[i],
      epochs = 50,
      validation_data = list(x_val_set, y_val_set[,c(1:2)]),
      class_weight = list("0"=1,"1"=cw)
    )
    
    val_loss[i,fold]<-min(cnn_history$metrics$val_loss)
    best_epoch_loss[i,fold]<-which(cnn_history$metrics$val_loss==min(cnn_history$metrics$val_loss))
    val_auc[i,fold] <- max(cnn_history$metrics$val_auc)
    best_epoch_auc[i,fold]<-which(cnn_history$metrics$val_auc==max(cnn_history$metrics$val_auc))
    
    print(i)
    print(fold)
    
  }
}

## ---- Find the best parameters ----
which(rowMeans(val_loss)==min(rowMeans(val_loss)))
best_mean_val_loss=which(rowMeans(val_loss)==min(rowMeans(val_loss)))
mean(val_loss[best_mean_val_loss[1],])
mean(best_epoch_loss[best_mean_val_loss[1],])
val_loss[best_mean_val_loss]      
best_epoch_loss[best_mean_val_loss]