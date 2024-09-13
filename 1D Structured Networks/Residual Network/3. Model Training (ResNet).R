# Training the model and examining performance on test set

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(caret)

## ---- Train the final model ----

## Run Setup from "Hyperparameter Tuning (ResNet).R" 

# These are the best parameters found 
best_param=tibble(filters = 64, kernel_size = 3, leaky_relu = T, batch_normalization = F, batch_size = 500)

input_shape <- c(249,1)
set_random_seed(15)
inputs <- layer_input(shape = input_shape)

block_1_output <- inputs %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu)

# Adjust block_2_output to include the first convolutional layer for block_2
block_2_prep <- block_1_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization)

block_2_output <- block_2_prep %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_add(block_1_output)

# Introduce a skip from block_1_output to block_3_output
block_3_output <- block_2_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_add(block_2_output) %>%
  layer_add(block_1_output) # Adding block_1_output as a skip to block_3_output

# Continue from block_3_output to block_4_output
block_4_output <- block_3_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_add(block_3_output)

# Introduce a skip from block_3_output to block_5_output
block_5_output <- block_4_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_add(block_4_output) %>%
  layer_add(block_3_output) # Adding block_3_output as a skip to block_5_output

outputs <- block_5_output %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  conv_activation_layer(filters = best_param$filters, 
                        kernel_size = best_param$kernel_size, 
                        leaky_relu = best_param$leaky_relu) %>%
  add_batch_normalization(batch_normalization = best_param$batch_normalization) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten() %>%
  layer_dense(2, activation="sigmoid")

model <- keras_model(inputs, outputs)

model %>% compile(
  optimizer = optimizers$legacy$Adam(1e-4),
  loss = loss_categorical_crossentropy,
  metrics = c("accuracy", tf$keras$metrics$AUC())
)

# Fit model
resnet_history <- model %>% fit(
  x_train, dummy_y_train,
  batch_size = best_param$batch_size,
  epochs = 100,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = list("0"=1,"1"=cw),
  callbacks = callbacks
)

eval <- evaluate(model, (x_test), dummy_y_test)
preds<-predict(model, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT", "SMB"))
cm <- confusionMatrix(species.predictions,as.factor(test$species))
auc_value <- eval[grepl("auc", names(eval))]
metrics <- c(eval[['loss']], cm$byClass['Sensitivity'], cm$byClass['Specificity'], auc_value)

cm
print(metrics)

