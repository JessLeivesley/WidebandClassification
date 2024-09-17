# Training the model and examining performance on test set

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(caret)

## ---- Train the final model ----

## Run Setup from "Hyperparameter Tuning (CNN).R" 

# These are the best parameters found 
best_param=tibble(filters1 = 16,
                  filters2 = 8,
                  filters3 = 32,
                  filters4 = 32,
                  filters5 = 32,
                  kernel_size = 3,
                  dropout1 = 0.0,
                  dropout2 = 0.0,
                  dropout3 = 0.0,
                  dropout4 = 0.1,
                  dropout5 = 0.0,
                  batch_size = 1500) 

set_random_seed(15)

cnn = keras_model_sequential()

#model
cnn %>% 
    layer_conv_1d(input_shape=c(249,1),
                  filters = best_param$filters1,
                  kernel_size = best_param$kernel_size, 
                  activation = 'relu',
                  padding = 'same',
                  strides = 1) %>%
    layer_dropout(rate = best_param$dropout1)%>%
    layer_batch_normalization()%>%
    layer_conv_1d(filters = best_param$filters2,
                  kernel_size = best_param$kernel_size,  
                  activation = 'relu',
                  padding = 'same',
                  strides = 1) %>%
    layer_dropout(rate = best_param$dropout2) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_conv_1d(filters = best_param$filters3,
                  kernel_size = best_param$kernel_size,  
                  activation = 'relu',
                  padding = 'same',
                  strides = 1) %>%
    layer_dropout(rate = best_param$dropout3) %>%
    layer_batch_normalization() %>%
    layer_conv_1d(filters = best_param$filters4,
                  kernel_size = best_param$kernel_size, 
                  activation = 'relu',
                  padding = 'same',
                  strides = 1) %>%
    layer_dropout(rate = best_param$dropout4) %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_batch_normalization() %>%
    layer_conv_1d(filters = best_param$filters5,
                  kernel_size = best_param$kernel_size, 
                  activation = 'relu',
                  padding = 'same',
                  strides = 1) %>%
    layer_dropout(rate = best_param$dropout5) %>%
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
    batch_size = best_param$batch_size,
    epochs = 200,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = list("0"=1,"1"=cw),
    callbacks = callbacks
  )
  

  eval <- evaluate(cnn, (x_test), dummy_y_test)
  preds<-predict(cnn, x=x_test)
  
  species.predictions<-apply(preds,1,which.max)
  species.predictions<-as.factor(ifelse(species.predictions == 1, "LT", "SMB"))
  cm <- confusionMatrix(species.predictions,as.factor(test$species))
  auc_value <- eval[grepl("auc", names(eval))]
  metrics <- c(eval[['loss']], cm$byClass['Sensitivity'], cm$byClass['Specificity'], auc_value)
  
  cm
  print(metrics)
  