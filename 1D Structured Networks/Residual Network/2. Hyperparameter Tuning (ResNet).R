# Hyperparameter Tuning for ResNet

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(caret)

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

# for implementing early stopping
callbacks <- list(
  callback_early_stopping(
    # Stop training when `val_loss` is no longer improving
    monitor = "val_loss",
    # "no longer improving" being defined as "no better than 1e-2 less"
    min_delta = 1e-2,
    # "no longer improving" being further defined as "for at least 2 epochs"
    patience = 30,
    restore_best_weights = TRUE
  )
)

## ---- Grid Search ----
# create grid of parameter space we want to search
filters <- c(16, 32, 64)
kernel_size <- c(3, 5, 7)
leaky_relu <- c(T, F)
batch_normalization <- c(T, F)
batch_size <- c(500, 800, 1000, 1200, 1500)

# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full<-expand.grid(filters = filters, kernel_size = kernel_size, 
                              leaky_relu = leaky_relu, 
                              batch_normalization = batch_normalization,
                              batch_size = batch_size)

set.seed(15)
n_subset <- 20
x<-sample(1:nrow(grid.search.full), n_subset,replace=F)
grid.search.subset<-grid.search.full[x,]

val_loss<-rep(NA,20)
best_epoch_loss<-rep(NA,20)
val_auc<-rep(NA,20)

optimizers <- keras::keras$optimizers

## ---- Run Search ----
# Run in two groups for memory (faster this way)

cw<-summary(as.factor(dummy_y_train[,1]))[2]/summary(as.factor(dummy_y_train[,1]))[1]

for (i in 1:10){
  print(sprintf("Processing Model #%d", i))
  
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
  
  model %>% compile(
    optimizer = optimizers$legacy$Adam(1e-4),
    loss = loss_categorical_crossentropy,
    metrics = c("accuracy", tf$keras$metrics$AUC())
  )
  
  # Fit model (just resnet)
  resnet_history <- model %>% fit(
    x_train, dummy_y_train,
    batch_size = grid.search.subset$batch_size[i],
    epochs = 100,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = list("0"=1,"1"=cw),
    callbacks = callbacks
  )
  
  val_loss[i]<-min(resnet_history$metrics$val_loss)
  best_epoch_loss[i]<-which(resnet_history$metrics$val_loss==min(resnet_history$metrics$val_loss))
  val_auc[i]<-resnet_history$metrics$val_auc[best_epoch_loss[i]]
}


for (i in 11:20){
  print(sprintf("Processing Model #%d", i))
  
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
  
  model %>% compile(
    optimizer = optimizers$legacy$Adam(1e-4),
    loss = loss_categorical_crossentropy,
    metrics = c("accuracy", tf$keras$metrics$AUC())
  )
  
  # Fit model (just resnet)
  resnet_history <- model %>% fit(
    x_train, dummy_y_train,
    batch_size = grid.search.subset$batch_size[i],
    epochs = 100,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = list("0"=1,"1"=cw),
    callbacks = callbacks
  )
  
  val_loss[i]<-min(resnet_history$metrics$val_loss)
  best_epoch_loss[i]<-which(resnet_history$metrics$val_loss==min(resnet_history$metrics$val_loss))
  val_auc[i]<-resnet_history$metrics$val_auc[best_epoch_loss[i]]
}


## ---- Find the best parameters ----
which(val_loss==min(val_loss))
best_mean_val_loss=which(val_loss==min(val_loss))
val_loss[best_mean_val_loss]      
best_epoch_loss[best_mean_val_loss]