# Training the model and examining performance on test set (RNN)

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)

## ---- Fit the best model 5 times ----

# create a list to store each of the history in
model_history<-list()

best_param=tibble(regrate=0.00001, lstmunits=256, neuron1=128)

for(fold in 1:5){
x_train_set<-x_data_train[folds[[fold]],,]
y_train_set<-dummy_y_train[folds[[fold]],]

x_val_set<-x_data_train[-folds[[fold]],,]
y_val_set<-dummy_y_train[-folds[[fold]],]

cw<-summary(as.factor(y_train_set[,1]))[2]/summary(as.factor(y_train_set[,1]))[1]

set_random_seed(15)
rnn = keras_model_sequential() # initialize model
# our input layer
rnn %>%
  layer_lstm(input_shape=c(5,249),units = best_param$lstmunits) %>%
  layer_activation_leaky_relu()%>%
  layer_batch_normalization()%>%
  layer_dense(units = best_param$neuron1,activity_regularizer = regularizer_l2(l=best_param$regrate)) %>%
  layer_activation_leaky_relu()%>%
  # layer_dropout(rate = best_param$droprate)%>%
  # layer_dense(units = best_param$neuron2,activity_regularizer = regularizer_l2(l=best_param$regrate)) %>%
  # layer_activation_leaky_relu()%>%
  layer_dense(units = 2, activation = 'sigmoid')

rnn %>% compile(
  loss = loss_binary_crossentropy,
  optimizer = optimizer_adam(3e-4),
  metrics = c('accuracy', tf$keras$metrics$AUC()))

model_history[[fold]] <- rnn %>% fit(
  x_train_set, y_train_set,
  batch_size = 1000, 
  epochs = 50,
  validation_data = list(x_val_set,y_val_set),
  class_weight = list("0"=1,"1"=cw))
}

## evaluating the best model on test data ##
