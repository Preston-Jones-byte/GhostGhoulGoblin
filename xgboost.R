library(bonsai)
library(lightgbm)
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


boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
          set_engine("lightgbm")             %>% 
          set_mode("classification")  #or "xgboost" but lightgbm is faster

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)
## CV tune, finalize and predict here



# tune --------------------------------------------------------------------


tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 10) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats= 5)

## Run the CV
CV_results <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics = metric_set(accuracy)) #Or leave metrics NULL

bestTune <- CV_results %>%
  select_best(metric="accuracy")



# Finalize ----------------------------------------------------------------


finalboost_wf <- boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)


## Predict
nb_preds <- predict(finalboost_wf, new_data=test, type="class")



# kaggle submission -------------------------------------------------------


Preds <-  nb_preds %>%
  bind_cols(test) %>%
  rename(type =.pred_class) %>%
  select(id, type)

vroom_write(x=Preds, file= "xgboostGGG.csv", delim=",")
