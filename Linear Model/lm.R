# In this example we will cover:
# RMSE
# Split dataset
# Cross Validation
# 10 fold cross validation
# 5 fold cross validation
# 5 cross 5 fold cross validation
# Making predictions on new data

# install caret package

# install.packages("caret")
library(caret)

# caret package contains different datasets for practise

# We will use diamonds dataset in this example
df <- data(diamonds)

# Set seed
set.seed(42)

# Shuffle row indices: rows
rows <- sample(nrow(diamonds))

# Randomly order data
diamonds <- diamonds[rows,]

# Determine row to split on: split

split <- round(nrow(diamonds) * .80)
# Create train
train <- diamonds[1:split,]

# Create test
test <- diamonds[(split + 1):nrow(diamonds), ]

# Fit lm model on train: model
model <- lm(price ~ ., data = train)

# Predict on test: p
p <- predict(model, test)

# The reason for using predict() on test data is that, how well your model predicts on new data
# Reduce overfitting

# Compute errors: error
error <- p - test$price

# Calculate RMSE (Root Mean Squared Error)
sqrt(mean(error^2))

# Computing the error on the training set is risky because the model may overfit the data used to train it.

# Cross Validation

# Advantage of cross-validation over a single train/test split
# It gives you multiple estimates of out-of-sample error, rather than a single estimate.
# If all of your estimates give similar outputs, you can be more certain of the model's accuracy.
# If your estimates give different outputs, that tells you the model does not perform consistently and suggests a problem with it.

# Fit lm model using 10-fold CV: model
model <- train(
  price ~ ., diamonds,
  method = "lm",
  trControl = trainControl(
    method = "cv", number = 10,
    verboseIter = TRUE
  )
)

# You can specify the model. "lm" represents Linear Model, "rf" represents Random forest.
# method = "cv" represents cross validation, number represents number of k folds.
# Print model to console
model

# Above is the 10 fold cross validation.
# You can also use 5 fold cross validation by changing number = 5

# Fit lm model using 10-fold CV: model
model1 <- train(
  price ~ ., diamonds,
  method = "lm",
  trControl = trainControl(
    method = "cv", number = 5,
    verboseIter = TRUE
  )
)

# Print model to console
model1

# 5 x 5-fold cross-validation

# you could repeat your entire cross-validation procedure 5 times for greater confidence in your estimates of the model's out-of-sample accuracy.

# Add repeats = 5 in trainControl()


#Making predictions on new data
# After fitting a model with train(), you can simply call predict() with new data.

#predict(model, diamonds)


