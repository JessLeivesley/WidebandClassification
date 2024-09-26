# Hyperparameter tuning of thee and four hidden layer DNN

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)

## ---- Create grid of all the hyperparameters ----
## three layer random search
regrate<-c(1e-6,1e-5,1e-4)
dropout<-c(0,.1,.15,.2)
neuron1<-c(128,96,64)
neuron2<-c(64,48,32)
neuron3<-c(32,24,16,8)

# expand the grid so that every possible combination of the above parameters is present. 
grid.search.full<-expand.grid(regrate=regrate,dropout=dropout,neuron1=neuron1,neuron2=neuron2,neuron3=neuron3)

## ---- Take 20 models to search through ----
# randomly select 20 of these models to fit. 
set.seed(15)
x<-sample(1:nrow(grid.search.full),20,replace=F)
grid.search.subset<-grid.search.full[x,]


## ---- Cross validation of 3 hidden layer models ----
# Create vectors to store validation loss and best epoch in
val_loss_3layer<-rep(NA,20)
best_epoch_3layer<-rep(NA,20)
val_auc_3layer<-rep(NA,20)

## Do this for 1:10 then 11:20 because it takes too long on its own
cw<-summary(as.factor(dummy_y_train[,1]))[2]/summary(as.factor(dummy_y_train[,1]))[1]

# for implementing early stopping
callbacks <- list(
  callback_early_stopping(
    monitor = "val_loss",
    min_delta = 1e-2,
    patience = 15,
    restore_best_weights = TRUE
  )
)

# for using legacy optimizers which work better with newer Macs
optimizers <- keras::keras$optimizers

# Class weight calculation to account for imbalanced data
cw<-summary(as.factor(dummy_y_train[,1]))[2]/summary(as.factor(dummy_y_train[,1]))[1]


for(i in 1:10){
  print(sprintf("Processing Model #%d", i))
    set_random_seed(15)
    model1 <- keras_model_sequential()
    model1 %>%
      layer_dense(units = grid.search.subset$neuron1[i],
                  input_shape = c(249),
                  activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dropout(grid.search.subset$dropout[i])%>%
      layer_dense(units = grid.search.subset$neuron2[i],
                  activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dropout(grid.search.subset$dropout[i])%>%
      layer_dense(units = grid.search.subset$neuron3[i],
                  activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dense(units = 2, activation = "sigmoid")
    
    model1 %>% compile(
      loss = 'binary_crossentropy',
      optimizer =  optimizers$legacy$Adam(3e-4),
      metrics = c("accuracy", tf$keras$metrics$AUC()))
    
    history <- model1 %>% fit(
      x_train, dummy_y_train,
      batch_size = 500, 
      epochs = 125,
      validation_data = list(x_validate,dummy_y_val),
      class_weight = list("0"=1,"1"=cw),
      callbacks = callbacks)
    
    val_loss_3layer[i]<-min(history$metrics$val_loss)
    best_epoch_3layer[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
    val_auc_3layer[i]<-history$metrics$val_auc[best_epoch_3layer[i]]

  }

for(i in 11:20){
  print(sprintf("Processing Model #%d", i))
  set_random_seed(15)
  model1 <- keras_model_sequential()
  model1 %>%
    layer_dense(units = grid.search.subset$neuron1[i], input_shape = c(249),activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(grid.search.subset$dropout[i])%>%
    layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(grid.search.subset$dropout[i])%>%
    layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
    layer_activation_leaky_relu()%>%
    layer_dense(units = 2, activation = "sigmoid")
  
  model1 %>% compile(
    loss = 'binary_crossentropy',
    optimizer =  optimizers$legacy$Adam(3e-4),
    metrics = c("accuracy", tf$keras$metrics$AUC()))
  
  history <- model1 %>% fit(
    x_train, dummy_y_train,
    batch_size = 500, 
    epochs = 125,
    validation_data = list(x_validate,dummy_y_val),
    class_weight = list("0"=1,"1"=cw),
    callbacks = callbacks)
  
  val_loss_3layer[i]<-min(history$metrics$val_loss)
  best_epoch_3layer[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
  val_auc_3layer[i]<-history$metrics$val_auc[best_epoch_3layer[i]]
  
  }


## ---- Create grid of all the hyperparameters for 4 hidden layers ----
regrate<-c(1e-6,1e-5,1e-4)
dropout<-c(0,0.1,0.15,0.2)
neuron1<-c(128,96,64)
neuron2<-c(64,48,32)
neuron3<-c(32,24,16)
neuron4<-c(16,8,4)

grid.search.full<-expand.grid(regrate=regrate,dropout=dropout,neuron1=neuron1,neuron2=neuron2,neuron3=neuron3,neuron4=neuron4)

## ---- Take 20 models to search through (4 hidden layers) ----
set.seed(15)
x<-sample(1:nrow(grid.search.full),20,replace=F)
grid.search.subset<-grid.search.full[x,]

## ---- Cross validation of 4 hidden layer models ----
# Create vectors to store validation loss and best epoch in
val_loss_4layer<-rep(NA,20)
best_epoch_4layer<-rep(NA,20)
val_auc_4layer<-rep(NA,20)

## Do this for 1:10 then 11:20 because it takes too long on its own
for(i in 2:10){
  print(sprintf("Processing Model #%d", i))
    set_random_seed(15)
    model1 <- keras_model_sequential()
    model1 %>%
      layer_dense(units = grid.search.subset$neuron1[i], input_shape = c(249),activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dropout(grid.search.subset$dropout[i])%>%
      layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dropout(grid.search.subset$dropout[i])%>%
      layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dropout(grid.search.subset$dropout[i])%>%
      layer_dense(units = grid.search.subset$neuron4[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      layer_dense(units = 2, activation = "sigmoid")
    
    model1 %>% compile(
      loss = 'binary_crossentropy',
      optimizer =  optimizers$legacy$Adam(3e-4),
      metrics = c("accuracy", tf$keras$metrics$AUC()))
    
    history <- model1 %>% fit(
      x_train, dummy_y_train,
      batch_size = 500, 
      epochs = 125,
      validation_data = list(x_validate,dummy_y_val),
      class_weight = list("0"=1,"1"=cw),
      callbacks = callbacks)
    
    val_loss_4layer[i]<-min(history$metrics$val_loss)
    best_epoch_4layer[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
    val_auc_4layer[i]<-history$metrics$val_auc[best_epoch_4layer[i]]   
  
  }

for(i in 11:20){
  print(sprintf("Processing Model #%d", i))
  set_random_seed(15)
  model1 <- keras_model_sequential()
  model1 %>%
    layer_dense(units = grid.search.subset$neuron1[i], input_shape = c(249),activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(grid.search.subset$dropout[i])%>%
    layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(grid.search.subset$dropout[i])%>%
    layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
    layer_activation_leaky_relu()%>%
    layer_dropout(grid.search.subset$dropout[i])%>%
    layer_dense(units = grid.search.subset$neuron4[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
    layer_activation_leaky_relu()%>%
    layer_dense(units = 2, activation = "sigmoid")
  
  model1 %>% compile(
    loss = 'binary_crossentropy',
    optimizer =  optimizers$legacy$Adam(3e-4),
    metrics = c("accuracy", tf$keras$metrics$AUC()))
  
  history <- model1 %>% fit(
    x_train, dummy_y_train,
    batch_size = 500, 
    epochs = 125,
    validation_data = list(x_validate,dummy_y_val),
    class_weight = list("0"=1,"1"=cw),
    callbacks = callbacks)
  
  val_loss_4layer[i]<-min(history$metrics$val_loss)
  best_epoch_4layer[i]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
  val_auc_4layer[i]<-history$metrics$val_auc[best_epoch_4layer[i]]   
  
}

