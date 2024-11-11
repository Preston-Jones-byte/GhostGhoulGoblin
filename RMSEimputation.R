library(tidyverse)
library(tidymodels)
library(vroom)
library(ggplot2)
library(ggmosaic)


# Read in the Data --------------------------------------------------------

train <- vroom("train.csv")
missSet <- vroom("trainWithMissingValues.csv")
test <- vroom("test.csv")


# EDA ---------------------------------------------------------------------

ggplot(data=train, aes(x=type, y=bone_length)) +
        geom_boxplot()

ggplot(data=train) + geom_mosaic(aes(x=product(color), fill=type))

# Recipe ------------------------------------------------------------------

miss_recipe <- recipe(type ~ ., data = missSet) %>%
                  step_impute_bag(all_predictors(),
                                  impute_with=imp_vars(all_predictors()), 
                                  trees=50)

prep_rep <- prep(miss_recipe)  
imputedSet <- bake(prep_rep, new_data = missSet)  

# RMSE imputation ---------------------------------------------------------

rmse_vec(train[is.na(missSet)],
         imputedSet[is.na(missSet)])


