# Training the model and examining performance on test set

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)

## ---- Fit the best model 5 times ----

# create a list to store each of the history in
model_history<-list()

for(fold in 1:5){
x_train_set<-x_train_S[x_train_S[,253] != fold,]
y_train_set<-dummy_y_train_S[dummy_y_train_S[,3]!=fold,]

x_val_set<-x_train_S[x_train_S[,253] == fold,]
y_val_set<-dummy_y_train_S[dummy_y_train_S[,3]==fold,]

cw<-summary(as.factor(y_train_set[,1]))[2]/summary(as.factor(y_train_set[,1]))[1]

set_random_seed(15)
model1 <- keras_model_sequential()
model1 %>%
  layer_dense(units = 128, input_shape = c(252),activity_regularizer = regularizer_l2(l=1e-4)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(0.2)%>%
  layer_dense(units = 32,activity_regularizer = regularizer_l2(l=1e-4)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(0.2)%>%
  layer_dense(units = 32 ,activity_regularizer = regularizer_l2(l=1e-4)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(0.2)%>%
  layer_dense(units = 16 ,activity_regularizer = regularizer_l2(l=1e-4)) %>%
  layer_activation_leaky_relu()%>%
  layer_dense(units = 2, activation = "sigmoid")

model1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer =  optimizer_adam(3e-4),
  metrics = c('accuracy'))


model_history[[fold]] <- model1 %>% fit(
  x_train_set[,c(1:252)], y_train_set[,c(1:2)],
  batch_size = 500, 
  epochs = best_epoch_3layer[7,fold],
  validation_data = list(x_val_set[,c(1:252)],y_val_set[,c(1:2)]),
  class_weight = list("0"=1,"1"=cw))
}


## edit to extract test loss, test sensitivity, test specificity, test auc
plot(history)
which(history$metrics$val_loss==min(history$metrics$val_loss))

evaluate(model1, x_test, dummy_y_test) 

preds<-predict(model1, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT",
                                      "SMB"))
confusionMatrix(species.predictions,as.factor(test$species))