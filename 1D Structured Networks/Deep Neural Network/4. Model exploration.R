# Exploring important predictors & model output

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(shapviz)
library(kernelshap)


pred_fun <- function(mod, X) predict(mod, data.matrix(X), batch_size = 1e4, verbose = FALSE)
ks <- kernelshap(model1, x_train_S[1:100,c(1:252)], bg_X = x_train_S[101:150,c(1:252)], pred_fun = pred_fun)  
shaps <- shapviz(ks)
sv_importance(shaps, show_numbers = TRUE)
sv_importance(shaps, kind = "beeswarm")
imp<-sv_importance(shaps, kind = "no")

frequencies<-rownames(imp)
frequencies<-as.numeric(gsub('F','',frequencies))
imp<-cbind.data.frame(imp,frequencies)
ggplot()+
  geom_line(data=imp,aes(x=frequencies,y=Class_1))+
  xlab("Frequency")+
  ylab("Mean SHAP")+
  theme_classic()+
  ggtitle("Lake Trout")+
  theme(text=element_text(size=16))

ggplot()+
  geom_line(data=imp,aes(x=frequencies,y=Class_2))+
  xlab("Frequency")+
  ylab("Mean SHAP")+
  theme_classic()+
  ggtitle("Smallmouth Bass")+
  theme(text=element_text(size=16))

