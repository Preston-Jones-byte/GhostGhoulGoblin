library(vroom)
library(tidyverse)
library(tidymodels)
library(embed)
library(kernlab)
library(keras)

# Data --------------------------------------------------------------------

train <- vroom("train.csv")
test <- vroom("test.csv")
train$type = as.factor(train$type)



# Recipe -------------------------------------------------------------------

nn_recipe <- recipe(type ~ . , data= train) %>%
          update_role(id, new_role="id") %>%
          step_mutate(color = as.factor(color)) %>% 
          step_dummy(color) %>%  ## Turn color to factor then dummy encode color
          step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]


# keras -------------------------------------------------------------------

nn_model <- mlp(hidden_units = tune(), epochs = 50)  %>%  #or 100 or 250
            set_engine("keras") %>% #verbose = 0 prints off less
            set_mode("classification")

nn_wf <- workflow() %>%
          add_recipe(nn_recipe) %>%
          add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 80)),
                            levels= 3)

nn_resamples <- vfold_cv(train, v = 2)

tuned_nn <- nn_wf %>%
        tune_grid(resamples = nn_resamples, grid = nn_tuneGrid)


tuned_nn %>% collect_metrics() %>%
    filter(.metric=="accuracy") %>%
    ggplot(aes(x=hidden_units, y=mean)) + geom_line() +
    labs(x = "Hidden Units", y = "Accuracy", 
          title = "Neural Network Tuning Results")

ggsave("nnplot.pdf")

## CV tune, finalize and predict here and save results
## This takes a few min (10 on my laptop) so run it on becker if you want

best_params <- tuned_nn %>%
               select_best(metric = "accuracy")

# Finalize the workflow with the best parameters
final_wf <- finalize_workflow(nn_wf, best_params) %>% 
            fit(train)

nb_preds <- predict(final_wf, new_data= test, type= "class")

Preds <-  nb_preds %>%
  bind_cols(test) %>%
  rename(type =.pred_class) %>%
  select(id, type)

vroom_write(x=Preds, file= "kerasnn.csv", delim=",")
