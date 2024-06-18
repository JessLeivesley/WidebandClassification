# Training the model and examining performance on test set

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(caret)

## ---- Fit the best model 5 times ----

for(fold in 1:5){
  x_train_set<-x_train[x_train[,250] != fold,]
  y_train_set<-dummy_y_train_S[dummy_y_train_S[,3]!=fold,]
  
  x_val_set<-x_train[x_train[,250] == fold,]
  y_val_set<-dummy_y_train_S[dummy_y_train_S[,3]==fold,]
  

  
  cw<-summary(as.factor(y_train_set[,1]))[2]/summary(as.factor(y_train_set[,1]))[1]
  
  # Scaling
  scaler <- preProcess(x_train_set[,1:249], method = 'scale')
  x_train_set <- predict(scaler, x_train_set[,1:249])
  x_val_set <- predict(scaler, x_val_set[,1:249])
  
  
  
  # below need to be extracted and inputted as values so only need to change this line everytime we have new optimal values
  best_param=tibble(filters = 4, kernel_size = 7, multipliers1 = 1,
                    multipliers2 = 1, multipliers3 = 2, multipliers4 = 2,
                    batch_size = 1000, droprate=0,droprate2=0,
                    droprate3=0.1, droprate4=0.1, droprate5=0.15,
                    regularizer_weight = 0.001)
  # best loss: 16, 5, t, f, 1200
  # best epoch: 29
  # got acc 83.59 (82.13 balanced)
  
  set_random_seed(15)
  cnn = keras_model_sequential()
  #model
  cnn %>% 
    layer_conv_1d(input_shape=c(249,1), filters = best_param$filters, kernel_size = best_param$kernel_size, 
                  activation = 'relu', padding = 'same', strides = 1)%>%
    layer_dropout(rate = best_param$droprate)%>%
    layer_batch_normalization()%>%
    layer_conv_1d(filters = best_param$filters * best_param$multipliers1, kernel_size = best_param$kernel_size,  
                  activation = 'relu', padding = 'same', strides = 1)%>%
    layer_dropout(rate = best_param$droprate2)%>%
    layer_batch_normalization()%>%
    layer_max_pooling_1d(pool_size = 2)%>%
    layer_conv_1d(filters = best_param$filters * best_param$multipliers1 * best_param$multipliers2, kernel_size = best_param$kernel_size,  
                  activation = 'relu', padding = 'same', strides = 1)%>%
    layer_dropout(rate = best_param$droprate3)%>%
    layer_batch_normalization()%>%
    layer_conv_1d(filters = best_param$filters * best_param$multipliers1 * best_param$multipliers2 * best_param$multipliers3, kernel_size = best_param$kernel_size, 
                  activation = 'relu', padding = 'same', strides = 1)%>%
    layer_dropout(rate = best_param$droprate4)%>%
    layer_max_pooling_1d(pool_size = 2)%>%
    layer_batch_normalization()%>%
    layer_conv_1d(filters = best_param$filters * best_param$multipliers1 * best_param$multipliers2 * best_param$multipliers3 * best_param$multipliers4, kernel_size = best_param$kernel_size, 
                  activation = 'relu', padding = 'same', strides = 1)%>%
    layer_dropout(rate = best_param$droprate5)%>%
    layer_batch_normalization()%>%
    layer_flatten() %>%
    layer_dense(units = 2, activation="sigmoid")
  
  cnn %>% compile(
    optimizer = optimizer_adam(weight_decay = best_param$regularizer_weight),
    loss = loss_binary_crossentropy(),
    metrics = c("accuracy", tf$keras$metrics$AUC())
  )
  
  # Fit model (just resnet)
  cnn_history[[fold]] <- cnn %>% fit(
      x_train_set, y_train_set[,c(1:2)],
      batch_size = best_param$batch_size,
      epochs = 50,
      validation_data = list(x_val_set, y_val_set[,c(1:2)]),
      class_weight = list("0"=1,"1"=cw)
    )
}

## edit to extract test loss, test sensitivity, test specificity, test auc
plot(cnn_history)


evaluate(cnn, (x_test), dummy_y_test)
preds<-predict(cnn, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT", "SMB"))
confusionMatrix(species.predictions,as.factor(test$species))