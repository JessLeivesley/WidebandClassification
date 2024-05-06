# Cross validation of thee and four hidden layer DNN

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
x<-sample(1:324,20,replace=F)
grid.search.subset<-grid.search.full[x,]


## ---- Cross validation of 3 hidden layer models ----
# Create vectors to store validation loss and best epoch in
val_loss_3layer<-matrix(nrow=20,ncol=5)
best_epoch_3layer<-matrix(nrow=20,ncol=5)

## Do this for 1:10 then 11:20 cus it takes too long on its own

for(i in 1:20){
  for(fold in 1:5){
    x_train_set<-x_train_S[x_train_S[,253] != fold,]
    y_train_set<-dummy_y_train_S[dummy_y_train_S[,3]!=fold,]
    
    x_val_set<-x_train_S[x_train_S[,253] == fold,]
    y_val_set<-dummy_y_train_S[dummy_y_train_S[,3]==fold,]
    
    cw<-summary(as.factor(y_train_set[,1]))[2]/summary(as.factor(y_train_set[,1]))[1]
    ## You will need to edit the model here depending on the number of layers you are fitting
    set_random_seed(15)
    model1 <- keras_model_sequential()
    model1 %>%
      layer_dense(units = grid.search.subset$neuron1[i], input_shape = c(252),activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
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
      optimizer =  optimizer_adam(3e-4),
      metrics = c('accuracy'))
    
    history <- model1 %>% fit(
      x_train_set[,c(1:252)], y_train_set[,c(1:2)],
      batch_size = 500, 
      epochs = 200,
      validation_data = list(x_val_set[,c(1:252)],y_val_set[,c(1:2)]),
      class_weight = list("0"=1,"1"=cw))
    
    val_loss_3layer[i,fold]<-min(history$metrics$val_loss)
    best_epoch_3layer[i,fold]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
    
    print(i)
    print(fold)
  }}

# Summarise the validation loss for each of the 20 models
rowMeans(val_loss_3layer)

# Model XX has the lowest validation loss across all folds

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
x<-sample(1:324,20,replace=F)
grid.search.subset<-grid.search.full[x,]

## ---- Cross validation of 4 hidden layer models ----
# Create vectors to store validation loss and best epoch in
val_loss_4layer<-matrix(nrow=20,ncol=5)
best_epoch_4layer<-matrix(nrow=20,ncol=5)

for(i in 1:20){
  for(fold in 1:5){
    x_train_set<-x_train_S[x_train_S[,253] != fold,]
    y_train_set<-dummy_y_train_S[dummy_y_train_S[,3]!=fold,]
    
    x_val_set<-x_train_S[x_train_S[,253] == fold,]
    y_val_set<-dummy_y_train_S[dummy_y_train_S[,3]==fold,]
    
    cw<-summary(as.factor(y_train_set[,1]))[2]/summary(as.factor(y_train_set[,1]))[1]
    
    set_random_seed(15)
    model1 <- keras_model_sequential()
    model1 %>%
      layer_dense(units = grid.search.subset$neuron1[i], input_shape = c(252),activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
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
      optimizer =  optimizer_adam(3e-4),
      metrics = c('accuracy'))
    
    history <- model1 %>% fit(
      x_train_set[,c(1:252)], y_train_set[,c(1:2)],
      batch_size = 500, 
      epochs = 100,
      validation_data = list(x_val_set[,c(1:252)],y_val_set[,c(1:2)]),
      class_weight = list("0"=1,"1"=cw))
    
    val_loss_4layer[i,fold]<-min(history$metrics$val_loss)
    best_epoch_4layer[i,fold]<-which(history$metrics$val_loss==min(history$metrics$val_loss))
  }}

# Summarise the validation loss for each of the 20 models
rowMeans(val_loss_4layer)

# Model XX has the lowest validation loss across all folds