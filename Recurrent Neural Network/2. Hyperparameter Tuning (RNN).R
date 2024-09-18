# Hyperparameter tuning (RNN)

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)

## ---- Set Up ----
# Class weight calculation to account for imbalanced data
cw<-summary(as.factor(dummy_y_train[,1]))[2]/summary(as.factor(dummy_y_train[,1]))[1]

# for implementing early stopping
callbacks <- list(
  callback_early_stopping(
    monitor = "val_loss",
    min_delta = 1e-2,
    patience = 25,
    restore_best_weights = TRUE
  )
)

# for using legacy optimizers which work better with newer Macs
optimizers <- keras::keras$optimizers


## ---- Create grid of all the hyperparameters ----

## create grid of parameter space we want to search
regrate<-c(1e-6,1e-5,1e-4)
lstmunits<-c(256,128,64)
neuron1<-c(256,128,64,32,16)
batchsize<-c(100, 500, 1000, 1500)

# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full<-expand.grid(regrate=regrate,lstmunits=lstmunits,neuron1=neuron1, batchsize = batchsize)

# randomly select 20 of these models to fit. 
set.seed(15)

x<-sample(1:nrow(grid.search.full),20,replace=F)
grid.search.subset<-grid.search.full[x,]

# objects to store evaluation metrics
val_loss<-vector()
best_epoch<-vector()
val_auc<-vector()

## Do this for 1:10 then 11:20 because it takes too long on its own
for(i in 1:10){ 
  print(sprintf("Processing Model #%d", i))
    set_random_seed(15)
    rnn = keras_model_sequential() # initialize model
    # our input layer
    rnn %>%
      layer_lstm(input_shape=c(5,249),units = grid.search.subset$lstmunits[i]) %>%
      layer_activation_leaky_relu()%>%
      layer_batch_normalization()%>%
      layer_dense(units = grid.search.subset$neuron1[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dense(units = 2, activation = 'sigmoid')
    
    rnn %>% compile(
      optimizer = optimizers$legacy$Adam(1e-4),
      loss = loss_binary_crossentropy(),
      metrics = c("accuracy", tf$keras$metrics$AUC())
      )
    
    rnn_history <- rnn %>% fit(
      x_data_train, dummy_y_train,
      batch_size = grid.search.subset$batchsize[i], 
      epochs = 200,
      validation_data = list(x_data_val,dummy_y_val),
      class_weight = list("0"=1,"1"=cw),
      callbacks = callbacks)
    
    val_loss[i]<-min(rnn_history$metrics$val_loss)
    best_epoch[i]<-which(rnn_history$metrics$val_loss==min(rnn_history$metrics$val_loss))[1]
    val_auc[i]<-max(rnn_history$metrics$val_auc)
}

for(i in 11:20){ 
  print(sprintf("Processing Model #%d", i))
  set_random_seed(15)
  rnn = keras_model_sequential() # initialize model
  # our input layer
  rnn %>%
    layer_lstm(input_shape=c(5,249),units = grid.search.subset$lstmunits[i]) %>%
    layer_activation_leaky_relu()%>%
    layer_batch_normalization()%>%
    layer_dense(units = grid.search.subset$neuron1[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
    layer_activation_leaky_relu()%>%
    layer_dense(units = 2, activation = 'sigmoid')
  
  rnn %>% compile(
    optimizer = optimizers$legacy$Adam(1e-4),
    loss = loss_binary_crossentropy(),
    metrics = c("accuracy", tf$keras$metrics$AUC())
  )
  
  rnn_history <- rnn %>% fit(
    x_data_train, dummy_y_train,
    batch_size = grid.search.subset$batchsize[i], 
    epochs = 200,
    validation_data = list(x_data_val,dummy_y_val),
    class_weight = list("0"=1,"1"=cw),
    callbacks = callbacks)
  
  val_loss[i]<-min(rnn_history$metrics$val_loss)
  best_epoch[i]<-which(rnn_history$metrics$val_loss==min(rnn_history$metrics$val_loss))[1]
  val_auc[i]<-max(rnn_history$metrics$val_auc)
}


## ---- Find the best parameters ----
which(val_loss==min(val_loss))
best_mean_val_loss=which(val_loss==min(val_loss))
val_loss[best_mean_val_loss]      
best_epoch[best_mean_val_loss]