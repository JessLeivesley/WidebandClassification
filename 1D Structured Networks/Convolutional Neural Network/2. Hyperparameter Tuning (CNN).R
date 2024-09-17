# Hyperparameter Tuning for CNN

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(caret)
library(rsample)
library(keras)

## ---- Set Up ----
# Class weight calculation to account for imbalanced data
cw<-summary(as.factor(dummy_y_train[,1]))[2]/summary(as.factor(dummy_y_train[,1]))[1]

# for implementing early stopping
callbacks <- list(
  callback_early_stopping(
    monitor = "val_loss",
    min_delta = 1e-2,
    patience = 50,
    restore_best_weights = TRUE
  )
)

# for using legacy optimizers which work better with newer Macs
optimizers <- keras::keras$optimizers

## ---- Grid Search ----
# create grid of parameter space we want to search
filters1 <- c(8, 16, 32, 64)
filters2 <- c(8, 16, 32, 64)
filters3 <- c(8, 16, 32, 64)
filters4 <- c(8, 16, 32, 64)
filters5 <- c(8, 16, 32, 64)
kernel_size <- c(3, 5, 7)
batch_size <- c(100, 500, 1000, 1500)
droprate1=c(0,0.1) 
droprate2=c(0,0.1) 
droprate3=c(0,0.1) 
droprate4=c(0,0.1) 
droprate5=c(0,0.1) 

# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full<-expand.grid(filters1 = filters1, filters2 = filters2, filters3 = filters3, filters4 = filters4, filters5 = filters5,kernel_size = kernel_size,
                              batch_size = batch_size, droprate=droprate1,droprate2=droprate2,
                              droprate3=droprate3, droprate4=droprate4, droprate5=droprate5)

set.seed(15)
n_subset <- 20
x<-sample(1:nrow(grid.search.full), n_subset,replace=F)
grid.search.subset<-grid.search.full[x,]

val_loss<-rep(NA,20)
best_epoch_loss<-rep(NA,20)
val_auc<-rep(NA,20)

## ---- Run Search ----
# Run in two groups for memory (faster this way)

for (i in 1:10){
    print(sprintf("Processing Model #%d", i))

    set_random_seed(15)
    
    cnn = keras_model_sequential()
    #model
    cnn %>% 
      layer_conv_1d(input_shape=c(249,1),
                    filters = grid.search.subset$filters1[i],
                    kernel_size = grid.search.subset$kernel_size[i], 
                    activation = 'relu',
                    padding = 'same',
                    strides = 1) %>%
      layer_dropout(rate = grid.search.subset$droprate[i])%>%
      layer_batch_normalization()%>%
      layer_conv_1d(filters = grid.search.subset$filters2[i],
                    kernel_size = grid.search.subset$kernel_size[i],  
                    activation = 'relu',
                    padding = 'same',
                    strides = 1) %>%
      layer_dropout(rate = grid.search.subset$droprate2[i]) %>%
      layer_batch_normalization() %>%
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_conv_1d(filters = grid.search.subset$filters3[i],
                    kernel_size = grid.search.subset$kernel_size[i],  
                    activation = 'relu',
                    padding = 'same',
                    strides = 1) %>%
      layer_dropout(rate = grid.search.subset$droprate3[i]) %>%
      layer_batch_normalization() %>%
      layer_conv_1d(filters = grid.search.subset$filters4[i],
                    kernel_size = grid.search.subset$kernel_size[i], 
                    activation = 'relu',
                    padding = 'same',
                    strides = 1) %>%
      layer_dropout(rate = grid.search.subset$droprate4[i]) %>%
      layer_max_pooling_1d(pool_size = 2) %>%
      layer_batch_normalization() %>%
      layer_conv_1d(filters = grid.search.subset$filters5[i],
                    kernel_size = grid.search.subset$kernel_size[i], 
                    activation = 'relu',
                    padding = 'same',
                    strides = 1) %>%
      layer_dropout(rate = grid.search.subset$droprate5[i]) %>%
      layer_batch_normalization() %>%
      layer_flatten() %>%
      layer_dense(units = 2, activation="sigmoid")
  
    cnn %>% compile(
      optimizer = optimizers$legacy$Adam(1e-4),
      loss = loss_binary_crossentropy(),
      metrics = c("accuracy", tf$keras$metrics$AUC())
    )
    
    
    # Fit model
    cnn_history <- cnn %>% fit(
      x_train, dummy_y_train,
      batch_size = grid.search.subset$batch_size[i],
      epochs = 200,
      validation_data = list(x_validate, dummy_y_val),
      class_weight = list("0"=1,"1"=cw),
      callbacks = callbacks
    )
    
    val_loss[i]<-min(cnn_history$metrics$val_loss)
    best_epoch_loss[i]<-which(cnn_history$metrics$val_loss==min(cnn_history$metrics$val_loss))
    val_auc[i] <- max(cnn_history$metrics$val_auc)
  
  }

for (i in 16:20){
  print(sprintf("Processing Model #%d", i))
  
  set_random_seed(15)
  
  cnn = keras_model_sequential()
  #model
  cnn %>% 
    layer_conv_1d(input_shape=c(249,1),
                  filters = grid.search.subset$filters1[i],
                  kernel_size = grid.search.subset$kernel_size[i], 
                  activation = 'relu',
                  padding = 'same',
                  strides = 1) %>%
    layer_dropout(rate = grid.search.subset$droprate[i])%>%
    layer_batch_normalization()%>%
    layer_conv_1d(filters = grid.search.subset$filters2[i],
                  kernel_size = grid.search.subset$kernel_size[i],  
                  activation = 'relu',
                  padding = 'same',
                  strides = 1) %>%
    layer_dropout(rate = grid.search.subset$droprate2[i]) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_conv_1d(filters = grid.search.subset$filters3[i],
                  kernel_size = grid.search.subset$kernel_size[i],  
                  activation = 'relu',
                  padding = 'same',
                  strides = 1) %>%
    layer_dropout(rate = grid.search.subset$droprate3[i]) %>%
    layer_batch_normalization() %>%
    layer_conv_1d(filters = grid.search.subset$filters4[i],
                  kernel_size = grid.search.subset$kernel_size[i], 
                  activation = 'relu',
                  padding = 'same',
                  strides = 1) %>%
    layer_dropout(rate = grid.search.subset$droprate4[i]) %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_batch_normalization() %>%
    layer_conv_1d(filters = grid.search.subset$filters5[i],
                  kernel_size = grid.search.subset$kernel_size[i], 
                  activation = 'relu',
                  padding = 'same',
                  strides = 1) %>%
    layer_dropout(rate = grid.search.subset$droprate5[i]) %>%
    layer_batch_normalization() %>%
    layer_flatten() %>%
    layer_dense(units = 2, activation="sigmoid")
  
  cnn %>% compile(
    optimizer = optimizers$legacy$Adam(1e-4),
    loss = loss_binary_crossentropy(),
    metrics = c("accuracy", tf$keras$metrics$AUC())
  )
  
  
  # Fit model
  cnn_history <- cnn %>% fit(
    x_train, dummy_y_train,
    batch_size = grid.search.subset$batch_size[i],
    epochs = 200,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = list("0"=1,"1"=cw),
    callbacks = callbacks
  )
  
  val_loss[i]<-min(cnn_history$metrics$val_loss)
  best_epoch_loss[i]<-which(cnn_history$metrics$val_loss==min(cnn_history$metrics$val_loss))
  val_auc[i] <- max(cnn_history$metrics$val_auc)
  
}

## ---- Find the best parameters ----
which(val_loss==min(val_loss))
best_mean_val_loss=which(val_loss==min(val_loss))
val_loss[best_mean_val_loss]      
best_epoch_loss[best_mean_val_loss]