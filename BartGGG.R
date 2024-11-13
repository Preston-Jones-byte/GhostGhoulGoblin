library(bonsai)
library(tidyverse)
library(tidymodels)
library(vroom)
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


# bart --------------------------------------------------------------------


bart_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate9
  set_engine("dbarts") %>% # might need to install10
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)
## CV tune, finalize and predict here



# tune --------------------------------------------------------------------


tuning_grid <- grid_regular(trees(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats= 5)

## Run the CV
CV_results <- bart_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics = metric_set(accuracy)) #Or leave metrics NULL

bestTune <- CV_results %>%
  select_best(metric="accuracy")



# Finalize ----------------------------------------------------------------


finalbart_wf <- bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)


## Predict
nb_preds <- predict(finalbart_wf, new_data=test, type="class")



# kaggle submission -------------------------------------------------------


Preds <-  nb_preds %>%
  bind_cols(test) %>%
  rename(type =.pred_class) %>%
  select(id, type)

vroom_write(x=Preds, file= "bartGGG.csv", delim=",")
