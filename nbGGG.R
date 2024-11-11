library(tidyverse)
library(tidymodels)
library(vroom)
library(naivebayes)
library(discrim)

# Data --------------------------------------------------------------------

train <- vroom("train.csv")
test <- vroom("test.csv")
train$type = as.factor(train$type)


# EDA ---------------------------------------------------------------------

dplyr::glimpse(train)



# Recipe ------------------------------------------------------------------


my_recipe <- recipe(type ~ ., data = train) %>%
            step_mutate(id, features = id)  %>%
            step_mutate(color = as.factor(color))

prep_rep <- prep(my_recipe)
bake(prep_rep, new_data = train)
            
            
            



# Naive bayes -------------------------------------------------------------------

## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naiveb

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# Tune smoothness and Laplace here
## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 10) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats= 3)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics = NULL) #Or leave metrics NULL

bestTune <- CV_results %>%
  select_best(metric="roc_auc")

finalnb_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)


## Predict
nb_preds <- predict(finalnb_wf, new_data=test, type="class")

Preds <-  nb_preds %>%
  bind_cols(test) %>%
  rename(type =.pred_class) %>%
  select(id, type)

vroom_write(x=Preds, file= "naivebayes.csv", delim=",")
