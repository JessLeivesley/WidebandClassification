# Hyperparameter Tuning for ResNet

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)

## ---- Data set up ----
# Remove the positional variables from the data sets generated in 1. Data Generation

x_train <- as.data.frame(x_train_S)%>%select(-aspectAngle, -Angle_minor_axis,-Angle_major_axis)
x_test <- as.data.frame(x_test)%>%select(-aspectAngle, -Angle_minor_axis,-Angle_major_axis)

x_train<-as.matrix(x_train)
x_test<-as.matrix(x_test)

## ---- Set up ----
# functions for adding layers conditionally
conv_activation_layer <- function(input_layer, filters, kernel_size, leaky_relu) {
  if (!leaky_relu) {
    # conv layer with ReLU activation
    output_layer <- input_layer %>% 
      layer_conv_1d(filters = filters, kernel_size = kernel_size, 
                    activation = 'relu', padding = 'same', strides = 1)
  } else {
    # conv layer followed by Leaky ReLU
    output_layer <- input_layer %>% 
      layer_conv_1d(filters = filters, kernel_size = kernel_size, 
                    padding = 'same', strides = 1) %>%
      layer_activation_leaky_relu()
  }
  return(output_layer)
}

add_batch_normalization <- function(input_layer, batch_normalization) {
  if (batch_normalization) {
    output_layer <- input_layer %>% layer_batch_normalization()
  } else {
    output_layer <- input_layer
  }
  return(output_layer)
}

## ---- Grid Search ----
# create grid of parameter space we want to search
filters <- c(16, 32, 64)
kernel_size <- c(3, 5, 7)
leaky_relu <- c(T, F)
batch_normalization <- c(F)
batch_size <- c(1000)

# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full<-expand.grid(filters = filters, kernel_size = kernel_size, 
                              leaky_relu = leaky_relu, 
                              batch_normalization = batch_normalization,
                              batch_size = batch_size)

set.seed(15)
# x<-sample(1:45,20,replace=F) # 45 = size of grid search (num of row)
x<-sample(1:nrow(grid.search.full),nrow(grid.search.full),replace=F)
grid.search.subset<-grid.search.full[x,]

## ---- Run Search ----
for (i in 1:nrow(grid.search.full)){
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
    
    input_shape <- c(249,1)
    set_random_seed(15)
    inputs <- layer_input(shape = input_shape)
    
    block_1_output <- inputs %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) # self-defined function with optional leaky relu (or relu)
    
    # Adjust block_2_output to include the first convolutional layer for block_2
    block_2_prep <- block_1_output %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      add_batch_normalization(batch_normalization = grid.search.subset$batch_normalization[i])
    
    block_2_output <- block_2_prep %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      add_batch_normalization(batch_normalization = grid.search.subset$batch_normalization[i]) %>%
      layer_add(block_1_output)
    
    # Introduce a skip from block_1_output to block_3_output
    block_3_output <- block_2_output %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      add_batch_normalization(batch_normalization = grid.search.subset$batch_normalization[i]) %>%
      layer_add(block_2_output) %>%
      layer_add(block_1_output) # Adding block_1_output as a skip to block_3_output
    
    # Continue from block_3_output to block_4_output
    block_4_output <- block_3_output %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      add_batch_normalization(batch_normalization = grid.search.subset$batch_normalization[i]) %>%
      layer_add(block_3_output)
    
    # Introduce a skip from block_3_output to block_5_output
    block_5_output <- block_4_output %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      add_batch_normalization(batch_normalization = grid.search.subset$batch_normalization[i]) %>%
      layer_add(block_4_output) %>%
      layer_add(block_3_output) # Adding block_3_output as a skip to block_5_output
    
    outputs <- block_5_output %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      conv_activation_layer(filters = grid.search.subset$filters[i], 
                            kernel_size = grid.search.subset$kernel_size[i], 
                            leaky_relu = grid.search.subset$leaky_relu[i]) %>%
      add_batch_normalization(batch_normalization = grid.search.subset$batch_normalization[i]) %>%
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_flatten() %>%
      layer_dense(2, activation="sigmoid")
    
    model <- keras_model(inputs, outputs)
    #model
    #plot(model,show_shapes = T)
    
    model %>% compile(
      optimizer = optimizer_adam(),
      loss = loss_categorical_crossentropy,
      metrics = c("accuracy", tf$keras$metrics$AUC())
    )
    
    # Fit model (just resnet)
    resnet_history <- model %>% fit(
      x_train_set, y_train_set[,c(1:2)],
      batch_size = grid.search.subset$batch_size[i],
      epochs = 75,
      validation_data = list(x_val_set, y_val_set[,c(1:2)])
    )
    
    val_loss[i,fold]<-min(resnet_history$metrics$val_loss)
    best_epoch_loss[i,fold]<-which(resnet_history$metrics$val_loss==min(resnet_history$metrics$val_loss))
    val_auc[i,fold] <- max(resnet_history$metrics$val_auc)
    best_epoch_auc[i,fold]<-which(resnet_history$metrics$val_auc==max(resnet_history$metrics$val_auc))
    
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
