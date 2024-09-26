# Training the model and examining performance on test set

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(caret)
library(shapviz)
library(kernelshap)

## ---- Train the final model ----
# These are the best parameters found 
best_param=tibble(neuron1 = 128, neuron2 = 64, neuron3 = 32, neuron4 = 4, dropout = 0.1, regrate=1e-6)


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

set_random_seed(15)

model1 <- keras_model_sequential()
model1 %>%
  layer_dense(units = best_param$neuron1, input_shape = c(249),activity_regularizer = regularizer_l2(l=best_param$regrate)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(best_param$dropout)%>%
  layer_dense(units = best_param$neuron2,activity_regularizer = regularizer_l2(l=best_param$regrate)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(best_param$dropout)%>%
  layer_dense(units = best_param$neuron3,activity_regularizer = regularizer_l2(l=best_param$regrate)) %>%
  layer_activation_leaky_relu()%>%
  layer_dropout(best_param$dropout)%>%
  layer_dense(units = best_param$neuron4,activity_regularizer = regularizer_l2(l=best_param$regrate)) %>%
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

eval <- evaluate(model1, (x_test), dummy_y_test)
preds<-predict(model1, x=x_test)

species.predictions<-apply(preds,1,which.max)
species.predictions<-as.factor(ifelse(species.predictions == 1, "LT", "SMB"))
cm <- confusionMatrix(species.predictions,as.factor(test$species))
auc_value <- eval[grepl("auc", names(eval))]
metrics <- c(eval[['loss']], cm$byClass['Sensitivity'], cm$byClass['Specificity'], auc_value)

cm
print(metrics)

# Exploring important predictors & model output
pred_fun <- function(mod, X) predict(mod, data.matrix(X), batch_size = 1e4, verbose = FALSE)
ks <- kernelshap(model1, x_train[1000:2000,c(1:249)], bg_X = x_train[1:200,c(1:249)], pred_fun = pred_fun)  
shaps <- shapviz(ks)
sv_importance(shaps, show_numbers = TRUE)
sv_importance(shaps, kind = "beeswarm")
imp<-sv_importance(shaps, kind = "no")

frequencies<-rownames(imp)
frequencies<-as.numeric(gsub('F','',frequencies))
imp<-cbind.data.frame(imp,frequencies)


# Plot the most important frequencies for each class
ggplot()+
  geom_bar(data=imp[order(imp$Class_1),],aes(x=factor(frequencies),y=Class_1),stat="identity",fill = "red",alpha=0.5,col="black")+
  scale_x_discrete(limits=c(as.factor(imp[order(imp$Class_1),][1:10,3])))+
  coord_flip()+
  theme_classic()+
  ggtitle("Lake Trout")+
  theme(text=element_text(size=16))+
  ylab("mean(|SHAP value|)")+
  xlab("Frequency")+
  ggplot()+
  geom_bar(data=imp[order(imp$Class_2),],aes(x=factor(frequencies),y=Class_2),stat="identity",fill = "blue",alpha=0.5,col="black")+
  scale_x_discrete(limits=c(as.factor(imp[order(imp$Class_2),][240:249,3])))+
  coord_flip()+
  theme_classic()+
  ggtitle("Smallmouth Bass")+
  theme(text=element_text(size=16))+
  ylab("mean(|SHAP value|)")+
  xlab("Frequency") 



# Plot the least important frequencies for each class
ggplot()+
  geom_bar(data=imp[order(imp$Class_1),],aes(x=factor(frequencies),y=Class_1),stat="identity",fill = "red",alpha=0.5,col="black")+
  scale_x_discrete(limits=c(as.factor(imp[order(imp$Class_1),][240:249,3])))+
  coord_flip()+
  theme_classic()+
  ggtitle("Lake Trout")+
  theme(text=element_text(size=16))+
  ylab("mean(|SHAP value|)")+
  xlab("Frequency")+
  ggplot()+
  geom_bar(data=imp[order(imp$Class_2),],aes(x=factor(frequencies),y=Class_2),stat="identity",fill = "blue",alpha=0.5,col="black")+
  scale_x_discrete(limits=c(as.factor(imp[order(imp$Class_2),][240:249,3])))+
  coord_flip()+
  theme_classic()+
  ggtitle("Smallmouth Bass")+
  theme(text=element_text(size=16))+
  ylab("mean(|SHAP value|)")+
  xlab("Frequency") 


