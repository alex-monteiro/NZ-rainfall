##########################################################################
###
### HarvardX's Data Science Project
### Capstone project: New Zealand Rainfall
### Data from 1960 to 2019
###
### Alexandre Monteiro
###
##########################################################################

## Libraries
library(tidyverse)
library(lubridate)
library(caret)
library(gam)

## NZ Rainfall dataset
## source: https://www.kaggle.com/pralabhpoudel/nz-rainfall-dataset
## Original data: https://www.stats.govt.nz/indicators/rainfall
##
## The datasets are licensed under a Creative Commons Attribution 4.0
## International License.
##
## What is measured
##
## This indicator measures annual and seasonal rainfall at 30 sites across
## New Zealand from 1960 to 2019. Precipitation is measured in milimiters.
##

# Load data
state_data <- read_csv("state_data.csv")

# The first few lines
head(state_data, n=10)

# Data adjustments
state_data <- state_data %>%
  mutate(year = year(period_end)) %>%
  select(-reference_period, -agent_number, -period_start, 
         -period_end, -lat, -lon) %>%
  select(site, season, year, precipitation, anomaly)

# Baseline
#
# Baseline is used to calculate the rainfall anommaly. Simply put, anomaly
# is the diference between the actual value of rainfall and a long term
# average. Our baseline will be the average precipitation for a period of
# 30 years, namely 1961 to 1990.
# We have to take into account that, for purposes of methodology, Summer
# is the period encompassed between December, 1st and March, 1st (open
# end) of the following year and is counted for that year, ie, from
# 1970-12-01 to 1971-02-28 the period is named Summer of 1971.
# That is why the filter is made using 'period_end' column.
# Although the data is avaiable as a column of the dataset, the
# calculation is performed for the sake of verification.
baseline <- state_data %>%
  filter(year >= 1961,
         year < 1991) %>%
  group_by(season, site) %>%
  summarize(base_precip = mean(precipitation))

# Now separate the state_data dataset into 'rainfall' and 'verification'
# datasets. The 'rainfall'dataset will be used to train the algorithm
# and will have 90% of the total data. The 'verification' dataset will
# be used in the final verification and will have the other 10% of the
# data.

# Setting the random seed
# if using R 3.5 or earlier, use `set.seed(2021)`
set.seed(2021, sample.kind = "Rounding") 

test_ind <- createDataPartition(y = state_data$precipitation,
                                times = 1, p = 0.1, list = FALSE)
rainfall <- state_data[-test_ind,]
verification <- state_data[test_ind,]

# Dividing the 'rainfall' dataset again into a 'train_set' and a
# 'test_set' that will be used during the algorithm training.
# The 'train_set'will have 90% of the data in the 'rainfall' dataset amd
# the 'test_set' will have the other 10% of the data in the 'rainfall'
# dataset.
test_ind <- createDataPartition(y = rainfall$precipitation,
                                times = 1, p = 0.1, list = FALSE)

train_set <- rainfall[-test_ind,]
test_set <- rainfall[test_ind,]

remove(test_ind)

# Data Exploration

# Data structure
str(rainfall)

# Data summary
summary(rainfall)

# Exploration
unique(rainfall$season)
unique(rainfall$site)

# Check for blank data
count_blank <- function(name) {
  sum(is.na(rainfall[, name]))
}

sapply(names(rainfall), count_blank)


# Mean precipitation values by season and site
rainfall %>%
  group_by(season, site) %>%
  summarize(mean_rainfall = mean(precipitation)) %>%
  pivot_wider(names_from = season,
              values_from = mean_rainfall)

# Annual anomaly data for all sites
rainfall %>%
  filter(season == "Annual") %>%
  ggplot(aes(x = year,
             y = precipitation)) +
  geom_line() +
  facet_wrap(. ~ site, ncol = 5) +
  ggtitle("Precipitation in New Zealand",
          subtitle = "Annual, 1960-2019")

# Take one station as an examble

# Get its baseline value
base_precip <- baseline %>%
  filter(site == "Milford Sound",
         season == "Summer") %>%
  pull(base_precip)

# Plot the graph
rainfall %>%
  filter(season == "Summer",
         site == "Milford Sound") %>%
  ggplot(aes(x = year, y = precipitation)) +
  geom_line() +
  geom_point() +
  geom_line(aes(y = mean(precipitation),
                lty = "mean precipitation"),
            col = "darkred",
            size = 1) +
  geom_line(aes(y = base_precip,
                lty = "baseline"),
            col = "darkgreen",
            size = 1) +
  ggtitle("Precipitations during summer in Milford Sound",
          subtitle = "1960-2019") +
  labs(linetype = NULL)

# Rain distribution
rainfall %>%
  ggplot(aes(x = precipitation,
             fill = season)) +
  geom_histogram(color = "black",
                 binwidth = 500,
                 position = "dodge") +
  ggtitle("Distribution of precipitations in the dataset")

rainfall %>%
  group_by(season) %>%
  summarize(mean = mean(precipitation),
            q75 = quantile(precipitation, 0.75))

# Anomaly distribution
# Rain distribution
rainfall %>%
  ggplot(aes(x = anomaly,
             fill = season)) +
  geom_histogram(color = "black",
                 binwidth = 500,
                 position = "dodge") +
  ggtitle("Distribution of anomaly in the dataset")


##
## Calculations
##

# Verify calculated anomaly accuracy
# Just for the sake of verification
rainfall %>%
  left_join(baseline, by = c("site", "season")) %>%
  mutate(calc_anom = precipitation - base_precip,
         diff = anomaly - calc_anom) %>%
  group_by(season) %>%
  summarize(error = sum(diff))

##
## Machine Learning Algorithms
##

# Our objective is to use machine learning algorythms to predict
# precipitation anomaly for each of the 30 climate stations.
0
# For each algorithm, the predictors will be:
# season, precipitation, year, and site.

#Set the random seed again.
set.seed(1973, sample.kind = "Rounding")

# Loss function: mean square error
RMSE <- function(true_values, predicted_values){
  sqrt(mean((true_values - predicted_values)^2))
}

#
# First algorithm: linear regression
#

# Model info
modelLookup("lm")

# Setting the control parameters
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3)

# Train the algorithm
lmFit <- train(anomaly ~ .,
               data = train_set,
               method = "lm",
               trControl = ctrl)

# Results
lmFit

# Summary
summary(lmFit)

# RMSE
lmPredict <- predict(lmFit,newdata = test_set)
error_value <- RMSE(test_set$anomaly, lmPredict)

# Add result to tabe
results_RMSE <- tibble(algorithm = "linear regression",
                       RMSE = error_value)

results_RMSE

# Plot predicted vs actual values in the train_set
data.frame(x = test_set$anomaly,
           y = lmPredict) %>%
  ggplot(aes(x = x,
             y = y)) +
  geom_point() +
  geom_abline(slope = 1) +
  ggtitle("Alogrithm: lm") +
  xlab("Actual data") +
  ylab("Predicted")

# Error analysis per season
test_set %>%
  cbind(fitted = lmPredict) %>%
  group_by(season) %>%
  summarize(rmse = RMSE(anomaly, fitted))

#
# Second algorithm: knn
#

# Model info
modelLookup("knn")

# Setting the control parameters
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3)

# Train the algorithm
knnFit <- train(anomaly ~ .,
                data = train_set,
                method = "knn",
                trControl = ctrl,
                tuneGrid = data.frame(k = seq(1,15,2)),
                preProcess = c("center", "scale"),
                tuneLength = 20)

# Results
knnFit

# Summary
summary(knnFit)

# Best number of neighbors
knnFit %>%
  ggplot(aes(x = .$results[k],
             y = .$results[RMSE])) +
  geom_point() +
  geom_line()

# Calculate the error
knnPredict <- predict(knnFit,newdata = test_set)
error_value <- RMSE(test_set$anomaly, knnPredict)

# Creater a table to group the resulst of all algorithms
results_RMSE <- results_RMSE %>%
  rbind(tibble(algorithm = "k-nearest neighbors",
               RMSE = error_value))

results_RMSE

# Plot predicted vs actual values in the train_set
data.frame(x = test_set$anomaly,
           y = knnPredict) %>%
ggplot(aes(x = x,
           y = y)) +
  geom_point() +
  geom_abline(slope = 1) +
  ggtitle("Alogrithm: knn") +
  xlab("Actual data") +
  ylab("Predicted")

# Error analysis per season
test_set %>%
  cbind(fitted = knnPredict) %>%
  group_by(season) %>%
  summarize(rmse = RMSE(anomaly, fitted))

#
# Third algorithm: gamLoess
#

# Model info
modelLookup("gamLoess")

# Setting the control parameters
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3)

grid <- expand.grid(span = seq(0.15, 0.65, len = 10),
                    degree = 1)

# Train the algorithm
gamFit <- train(anomaly ~ .,
                data = train_set,
                method = "gamLoess",
                trControl = ctrl,
                tuneGrid = grid)

# Results
gamFit

# RMSE
gamPredict <- predict(gamFit,newdata = test_set)
error_value <- RMSE(test_set$anomaly, gamPredict)

# Add result to table
results_RMSE <- results_RMSE %>%
  rbind(tibble(algorithm = "generalized aditive model",
               RMSE = error_value))
results_RMSE

# Plot predicted vs actual values in the train_set
data.frame(x = test_set$anomaly,
           y = gamPredict) %>%
  ggplot(aes(x = x,
             y = y)) +
  geom_point() +
  geom_abline(slope = 1) +
  ggtitle("Alogrithm: gamLoess") +
  xlab("Actual data") +
  ylab("Predicted")

# Error analysis per season
test_set %>%
  cbind(fitted = gamPredict) %>%
  group_by(season) %>%
  summarize(rmse = RMSE(anomaly, fitted))


#
# Fourth algorithm: regression tree
#

# Model info
modelLookup("rpart")

# Setting the control parameters
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3)
grid <- expand.grid(cp = range(0, 3, 0.1))

# Train the algorithm
rpFit <- train(anomaly ~ .,
               data = train_set,
               method = "rpart",
               trControl = ctrl,
               tuneGrid = grid)

# Results
rpFit

# Tree: general form
plot(rpFit$finalModel)

# RMSE
rpPredict <- predict(rpFit,newdata = test_set)
error_value <- RMSE(test_set$anomaly, rpPredict)

# Add result to tabe
results_RMSE <- results_RMSE %>%
  rbind(tibble(algorithm = "Regression Tree",
               RMSE = error_value))
results_RMSE

# Plot predicted vs actual values in the train_set
data.frame(x = test_set$anomaly,
           y = rpPredict) %>%
  ggplot(aes(x = x,
             y = y)) +
  geom_point() +
  geom_abline(slope = 1) +
  ggtitle("Alogrithm: regression tree") +
  xlab("Actual data") +
  ylab("Predicted")

# Error analysis per season
test_set %>%
  cbind(fitted = rpPredict) %>%
  group_by(season) %>%
  summarize(rmse = RMSE(anomaly, fitted))

#
# Fifth algorithm: Bayesian Regularized Neural Network
#

# Model info
modelLookup("brnn")

# Setting the control parameters
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3)
grid <- expand.grid(neurons = seq(2,3,1))

# Train the algorithm
brnnFit <- train(anomaly ~ .,
                 data = train_set,
                 method = "brnn",
                 trControl = ctrl,
                 tuneGrid = grid)


# Results
brnnFit

# RMSE
brnnPredict <- predict(brnnFit,newdata = test_set)
error_value <- RMSE(test_set$anomaly, brnnPredict)

# Add result to tabe
results_RMSE <- results_RMSE %>%
  rbind(tibble(algorithm = "Bayesian Regularized Neural Network",
               RMSE = error_value))
results_RMSE

# Plot predicted vs actual values in the train_set
data.frame(x = test_set$anomaly,
           y = brnnPredict) %>%
  ggplot(aes(x = x,
             y = y)) +
  geom_point() +
  geom_abline(slope = 1) +
  ggtitle("Alogrithm: brnn") +
  xlab("Actual data") +
  ylab("Predicted")

# Given brnn is, by far, the minimum RMSE among the chosen algorithms, we
# will run the final verification only for it.

# RMSE
brnnPredict <- predict(brnnFit,newdata = verification)
error_value <- RMSE(verification$anomaly, brnnPredict)

# Add result to tabe
results_RMSE <- results_RMSE %>%
  rbind(tibble(algorithm = "BRNN - Verification",
               RMSE = error_value))
results_RMSE


# Plot predicted vs actual values in the train_set
data.frame(x = verification$anomaly,
           y = brnnPredict) %>%
  ggplot(aes(x = x,
             y = y)) +
  geom_point() +
  geom_abline(slope = 1) +
  ggtitle("Alogrithm: brnn - Verification") +
  xlab("Actual data") +
  ylab("Predicted")