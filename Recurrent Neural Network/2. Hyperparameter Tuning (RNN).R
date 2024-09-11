# Hyperparameter tuning (RNN)

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)

## ---- One hidden layer ---- 
## ---- Create grid of all the hyperparameters ----

## create grid of parameter space we want to search
regrate<-c(1e-6,1e-5,1e-4)
lstmunits<-c(256,128,64)
neuron1<-c(256,128,64,32,16)

# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full<-expand.grid(regrate=regrate,lstmunits=lstmunits,neuron1=neuron1)

# randomly select 20 of these models to fit. 
set.seed(15)

x<-sample(1:nrow(grid.search.full),20,replace=F)
grid.search.subset<-grid.search.full[x,]

# objects to store evaluation metrics
val_loss1<-vector()
best_epoch_loss1<-vector()
val_auc1<-vector()
best_epoch_auc1<-vector()

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
      loss = loss_binary_crossentropy,
      optimizer = optimizer_adam(3e-4),
      metrics = c('accuracy', tf$keras$metrics$AUC()))
    
    history <- rnn %>% fit(
      x_data_train, dummy_y_train,
      batch_size = 1000, 
      epochs = 50,
      validation_data = list(x_data_val,dummy_y_val),
      class_weight = list("0"=1,"1"=2))
    
    
    val_loss1[i]<-min(history$metrics$val_loss)
    best_epoch_loss1[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))[1]
    val_auc1[i]<-max(history$metrics$val_auc)
    best_epoch_auc1[i]<-which(history$metrics$val_auc==max(history$metrics$val_auc))
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
      loss = loss_binary_crossentropy,
      optimizer = optimizer_adam(3e-4),
      metrics = c('accuracy', tf$keras$metrics$AUC()))
    
    history <- rnn %>% fit(
      x_data_train, dummy_y_train,
      batch_size = 1000, 
      epochs = 50,
      validation_data = list(x_data_val,dummy_y_val),
      class_weight = list("0"=1,"1"=2))
    
    
    val_loss1[i]<-min(history$metrics$val_loss)
    best_epoch_loss1[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))[1]
    val_auc1[i]<-max(history$metrics$val_auc)
    best_epoch_auc1[i]<-which(history$metrics$val_auc==max(history$metrics$val_auc))
}


## ---- Two hidden layers ---- 
## ---- Create grid of all the hyperparameters ----

## create grid of parameter space we want to search
regrate<-c(1e-6,1e-5,1e-4)
droprate=c(0,0.1,0.15) 
lstmunits<-c(256,128,64)
neuron1<-c(256,128,64)
neuron2=c(64,32,16)

# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full<-expand.grid(regrate=regrate,lstmunits=lstmunits,neuron1=neuron1,
                              droprate=droprate,neuron2=neuron2
)

# randomly select 20 of these models to fit. 
set.seed(15)

x<-sample(1:nrow(grid.search.full),20,replace=F)
grid.search.subset<-grid.search.full[x,]

# objects to store evaluation metrics
val_loss2<-vector()
best_epoch_loss2<-vector()
val_auc2<-vector()
best_epoch_auc2<-vector()

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
      layer_dropout(rate = grid.search.subset$droprate[i])%>%
      layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dense(units = 2, activation = 'sigmoid')
    
    rnn %>% compile(
      loss = loss_binary_crossentropy,
      optimizer = optimizer_adam(3e-4),
      metrics = c('accuracy', tf$keras$metrics$AUC()))
    
    history <- rnn %>% fit(
      x_data_train, dummy_y_train,
      batch_size = 1000, 
      epochs = 50,
      validation_data = list(x_data_val,dummy_y_val),
      class_weight = list("0"=1,"1"=2))
    
    
    val_loss2[i]<-min(history$metrics$val_loss)
    best_epoch_loss2[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))[1]
    val_auc2[i]<-max(history$metrics$val_auc)
    best_epoch_auc2[i]<-which(history$metrics$val_auc==max(history$metrics$val_auc))
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
      layer_dropout(rate = grid.search.subset$droprate[i])%>%
      layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dense(units = 2, activation = 'sigmoid')
    
    rnn %>% compile(
      loss = loss_binary_crossentropy,
      optimizer = optimizer_adam(3e-4),
      metrics = c('accuracy', tf$keras$metrics$AUC()))
    
    history <- rnn %>% fit(
      x_data_train, dummy_y_train,
      batch_size = 1000, 
      epochs = 50,
      validation_data = list(x_data_val,dummy_y_val),
      class_weight = list("0"=1,"1"=2))
    
    
    val_loss2[i]<-min(history$metrics$val_loss)
    best_epoch_loss2[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))[1]
    val_auc2[i]<-max(history$metrics$val_auc)
    best_epoch_auc2[i]<-which(history$metrics$val_auc==max(history$metrics$val_auc))
}


## ---- Three hidden layer ---- 
## ---- Create grid of all the hyperparameters ----

## create grid of parameter space we want to search
regrate<-c(1e-6,1e-5,1e-4)
droprate=c(0,0.1,0.15) # only needed in 2 hidden layer
lstmunits<-c(256,128,64)
neuron1<-c(256,128,64)
neuron2=c(64,32,16) # only needed in 2 hidden layers
neuron3=c(16,8,4) # only needed in 3 hidden layers

# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full<-expand.grid(regrate=regrate,lstmunits=lstmunits,neuron1=neuron1,
                              droprate=droprate,neuron2=neuron2,neuron3=neuron3
)

# randomly select 20 of these models to fit. 
set.seed(15)

x<-sample(1:nrow(grid.search.full),20,replace=F)
grid.search.subset<-grid.search.full[x,]

# objects to store evaluation metrics
val_loss3<-vector()
best_epoch_loss3<-vector()
val_auc3<-vector()
best_epoch_auc3<-vector()

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
      layer_dropout(rate = grid.search.subset$droprate[i])%>%
      layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dropout(rate = grid.search.subset$droprate[i])%>%
      layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dense(units = 2, activation = 'sigmoid')
    
    rnn %>% compile(
      loss = loss_binary_crossentropy,
      optimizer = optimizer_adam(3e-4),
      metrics = c('accuracy', tf$keras$metrics$AUC()))
    
    history <- rnn %>% fit(
      x_data_train, dummy_y_train,
      batch_size = 1000, 
      epochs = 50,
      validation_data = list(x_data_val,dummy_y_val),
      class_weight = list("0"=1,"1"=2))
    
    val_loss3[i]<-min(history$metrics$val_loss)
    best_epoch_loss3[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))[1]
    val_auc3[i]<-max(history$metrics$val_auc)
    best_epoch_auc3[i]<-which(history$metrics$val_auc==max(history$metrics$val_auc))
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
      layer_dropout(rate = grid.search.subset$droprate[i])%>%
      layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dropout(rate = grid.search.subset$droprate[i])%>%
      layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dense(units = 2, activation = 'sigmoid')
    
    rnn %>% compile(
      loss = loss_binary_crossentropy,
      optimizer = optimizer_adam(3e-4),
      metrics = c('accuracy', tf$keras$metrics$AUC()))
    
    history <- rnn %>% fit(
      x_data_train, dummy_y_train,
      batch_size = 1000, 
      epochs = 50,
      validation_data = list(x_data_val,dummy_y_val),
      class_weight = list("0"=1,"1"=2))
    
    val_loss3[i]<-min(history$metrics$val_loss)
    best_epoch_loss3[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))[1]
    val_auc3[i]<-max(history$metrics$val_auc)
    best_epoch_auc3[i]<-which(history$metrics$val_auc==max(history$metrics$val_auc))
}
