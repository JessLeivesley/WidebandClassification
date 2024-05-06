# Hyperparameter tuning

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)

## ---- One hidden layer ---- 
## ---- Create grid of all the hyperparameters ----

## create grid of parameter space we want to search
regrate<-c(1e-6,1e-5,1e-4)
# droprate=c(0,0.1,0.15) # only needed in 2 hidden layer
# droprate2=c(0,0.1,0.15) # only needed in 3 hidden layer if want different drop rates
lstmunits<-c(256,128,64)
neuron1<-c(256,128,64,32,16)
# neuron2=c(64,32,16) # only needed in 2 hidden layers
# neuron3=c(16,8,4) # only needed in 3 hidden layers

# expand the grid so that every possible combination of the above parameters is present. 
# creating every possible combination to test
grid.search.full<-expand.grid(regrate=regrate,lstmunits=lstmunits,neuron1=neuron1
                              # droprate=droprate,droprate2=droprate2,neuron2=neuron2,neuron3=neuron3
)

# randomly select 20 of these models to fit. 
set.seed(15)

x<-sample(1:nrow(grid.search.full),20,replace=F)
grid.search.subset<-grid.search.full[x,]

# objects to store evaluation metrics
val_loss<-matrix(nrow=20,ncol=5)
best_epoch_loss<-matrix(nrow=20,ncol=5)
val_auc<-matrix(nrow=20,ncol=5)
best_epoch_auc<-matrix(nrow=20,ncol=5)

## Do this for 1:10 then 11:20 because it takes too long on its own
for(i in 1:10){ 
  for(fold in 1:5){ 
    x_train_set<-x_data_train[folds[[fold]],,]
    y_train_set<-dummy_y_train[folds[[fold]],]
    
    x_val_set<-x_data_train[-folds[[fold]],,]
    y_val_set<-dummy_y_train[-folds[[fold]],]
    
    set_random_seed(15)
    rnn = keras_model_sequential() # initialize model
    # our input layer
    rnn %>%
      layer_lstm(input_shape=c(5,249),units = grid.search.subset$lstmunits[i]) %>%
      layer_activation_leaky_relu()%>%
      layer_batch_normalization()%>%
      layer_dense(units = grid.search.subset$neuron1[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      # layer_dropout(rate = grid.search.subset$droprate[i])%>%
      # layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      # layer_activation_leaky_relu()%>%
      # layer_dropout(rate = grid.search.subset$droprate2[i])%>%
      # layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      # layer_activation_leaky_relu()%>%
      layer_dense(units = 2, activation = 'sigmoid')
    
    rnn %>% compile(
      loss = loss_binary_crossentropy,
      optimizer = optimizer_adam(3e-4),
      metrics = c('accuracy', tf$keras$metrics$AUC()))
    
    history <- rnn %>% fit(
      x_train_set, y_train_set,
      batch_size = 1000, 
      epochs = 50,
      validation_data = list(x_val_set,y_val_set),
      class_weight = list("0"=1,"1"=2))
    
    
    val_loss[i,fold]<-min(history$metrics$val_loss)
    best_epoch_loss[i,fold]<-which(history$metrics$val_loss==min(history$metrics$val_loss))[1]
    val_auc[i,fold]<-max(history$metrics$val_auc)
    best_epoch_auc[i,fold]<-which(history$metrics$val_auc==max(history$metrics$val_auc))
    print(i)
    print(fold) 
  }
}

for(i in 11:20){ 
  for(fold in 1:5){ 
    x_train_set<-x_data_train[folds[[fold]],,]
    y_train_set<-dummy_y_train[folds[[fold]],]
    
    x_val_set<-x_data_train[-folds[[fold]],,]
    y_val_set<-dummy_y_train[-folds[[fold]],]
    
    set_random_seed(15)
    rnn = keras_model_sequential() # initialize model
    # our input layer
    rnn %>%
      layer_lstm(input_shape=c(5,249),units = grid.search.subset$lstmunits[i]) %>%
      layer_activation_leaky_relu()%>%
      layer_batch_normalization()%>%
      layer_dense(units = grid.search.subset$neuron1[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      layer_activation_leaky_relu()%>%
      # layer_dropout(rate = grid.search.subset$droprate[i])%>%
      # layer_dense(units = grid.search.subset$neuron2[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      # layer_activation_leaky_relu()%>%
      # layer_dropout(rate = grid.search.subset$droprate2[i])%>%
      # layer_dense(units = grid.search.subset$neuron3[i],activity_regularizer = regularizer_l2(l=grid.search.subset$regrate[i])) %>%
      # layer_activation_leaky_relu()%>%
      layer_dense(units = 2, activation = 'sigmoid')
    
    rnn %>% compile(
      loss = loss_binary_crossentropy,
      optimizer = optimizer_adam(3e-4),
      metrics = c('accuracy', tf$keras$metrics$AUC()))
    
    history <- rnn %>% fit(
      x_train_set, y_train_set,
      batch_size = 1000, 
      epochs = 50,
      validation_data = list(x_val_set,y_val_set),
      class_weight = list("0"=1,"1"=2))
    
    
    val_loss[i,fold]<-min(history$metrics$val_loss)
    best_epoch_loss[i,fold]<-which(history$metrics$val_loss==min(history$metrics$val_loss))[1]
    val_auc[i,fold]<-max(history$metrics$val_auc)
    best_epoch_auc[i,fold]<-which(history$metrics$val_auc==max(history$metrics$val_auc))
    print(i)
    print(fold) 
  }
}

## ---- Two hidden layers ---- 