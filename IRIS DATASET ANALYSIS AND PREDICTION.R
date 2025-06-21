# Load libraries
library(dplyr)
library(GGally)
library(ggplot2)
library(datasets)
library(caret)
library(randomForest)
library(e1071)
library(nnet)
library(rpart)
library(pROC)

# Load the Iris dataset
iris <- datasets::iris

# 1. Data Exploration
# View the dataset
View(iris)
head(iris, 5)
tail(iris, 5)

# Summary descriptive Stats
summary(iris)

# Check for missing values
sum(is.na(iris))

# 2. Descriptive Statistics by Species
# Sepal Length Statistics
sepal_length_stats <- iris %>% 
  group_by(Species) %>% summarize(
    mean_sepal_length = mean(Sepal.Length), 
    sd_sepal_length = sd(Sepal.Length), 
    min_sepal_length = min(Sepal.Length), 
    max_sepal_length = max(Sepal.Length)
  )
print(sepal_length_stats)

# Sepal Width Statistics
sepal_width_stats <- iris %>% 
  group_by(Species) %>% summarize(
    mean_sepal_width = mean(Sepal.Width), 
    sd_sepal_width = sd(Sepal.Width), 
    min_sepal_width = min(Sepal.Width), 
    max_sepal_width = max(Sepal.Width)
  )
print(sepal_width_stats)

# Petal Length Statistics
petal_length_stats <- iris %>% 
  group_by(Species) %>% summarize(
    mean_petal_length = mean(Petal.Length), 
    sd_petal_length = sd(Petal.Length), 
    min_petal_length = min(Petal.Length), 
    max_petal_length = max(Petal.Length)
  )
print(petal_length_stats)

# Petal Width Statistics
petal_width_stats <- iris %>% 
  group_by(Species) %>% summarize(
    mean_petal_width = mean(Petal.Width), 
    sd_petal_width = sd(Petal.Width), 
    min_petal_width = min(Petal.Width), 
    max_petal_width = max(Petal.Width)
  )
print(petal_width_stats)

# 3. Data Visualization
# Overall plot
plot(iris, col = "blue")

# Scatter Plots
# Sepal Width vs Sepal Length
ggplot(iris) +
  geom_point(aes(x=Sepal.Width, y=Sepal.Length), color="red") +
  labs(title = "Sepal Width vs Sepal Length")

# Petal Width vs Petal Length
ggplot(iris)+
  geom_point(aes(x=Petal.Width, y=Petal.Length), color="blue") +
  labs(title = "Petal Width vs Petal Length")

# Histograms
# Sepal Width
ggplot(iris) +
  geom_histogram(aes(x=Sepal.Width), fill = "purple") +
  labs(title = "Sepal Width")

# Sepal Length
ggplot(iris) +
  geom_histogram(aes(x=Sepal.Length), fill = "orange") +
  labs(title = "Sepal Length")

# Petal Width
ggplot(iris) +
  geom_histogram(aes(x=Petal.Width), fill = "cyan") +
  labs(title = "Petal Width")

# Petal Length
ggplot(iris) +
  geom_histogram(aes(x=Petal.Length), fill = "green") +
  labs(title = "Petal Length")

# Relationship Sepal and Species
ggplot(iris) +
  geom_point(aes(x=Sepal.Width, y=Sepal.Length, color=Species)) +
  facet_wrap(~Species) +
  labs(title = "Relationship Sepal and Species")

# Relationship Petal and Species
ggplot(iris) +
  geom_point(aes(x=Petal.Width, y=Petal.Length, color=Species)) +
  facet_wrap(~Species) +
  labs(title = "Relationship Petal and Species")

# Pair Plot
ggpairs(iris,
        columns = 1:4,
        aes(color = Species),
        upper = list(continuous = "points", combo = "box_no_facet"),
        legend = 2,
        title = "Relationship Between Features By Species"
)

# 4. Predictive Modeling
# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train_data <- iris[index, ]
test_data <- iris[-index, ]

# 4.1 Random Forest Classification
rf_model <- randomForest(Species ~ ., data = train_data, ntree = 100)
rf_predictions <- predict(rf_model, test_data)
rf_confusion_matrix <- confusionMatrix(rf_predictions, test_data$Species)

# 4.2 Support Vector Machine (SVM)
svm_model <- svm(Species ~ ., data = train_data)
svm_predictions <- predict(svm_model, test_data)
svm_confusion_matrix <- confusionMatrix(svm_predictions, test_data$Species)

# 4.3 Multinomial Logistic Regression
multinom_model <- multinom(Species ~ ., data = train_data)
multinom_predictions <- predict(multinom_model, test_data)
multinom_confusion_matrix <- confusionMatrix(multinom_predictions, test_data$Species)

# 4.4 Decision Tree
dt_model <- rpart(Species ~ ., data = train_data)
dt_predictions <- predict(dt_model, test_data, type = "class")
dt_confusion_matrix <- confusionMatrix(dt_predictions, test_data$Species)

# Print model performance
print("Random Forest Confusion Matrix:")
print(rf_confusion_matrix)

print("SVM Confusion Matrix:")
print(svm_confusion_matrix)

print("Multinomial Logistic Regression Confusion Matrix:")
print(multinom_confusion_matrix)

print("Decision Tree Confusion Matrix:")
print(dt_confusion_matrix)

# Cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Perform cross-validation for each model
rf_cv <- train(Species ~ ., data = iris, 
               method = "rf", 
               trControl = train_control)

svm_cv <- train(Species ~ ., data = iris, 
                method = "svmRadial", 
                trControl = train_control)

multinom_cv <- train(Species ~ ., data = iris, 
                     method = "multinom", 
                     trControl = train_control)

dt_cv <- train(Species ~ ., data = iris, 
               method = "rpart", 
               trControl = train_control)

# Compare cross-validation results
print("Cross-Validation Results:")
print(rf_cv)
print(svm_cv)
print(multinom_cv)
print(dt_cv)

# Feature Importance for Random Forest
importance_rf <- varImp(rf_model)
print("Random Forest Feature Importance:")
print(importance_rf)

# Visualization of predictions
results_df <- data.frame(
  Actual = test_data$Species,
  RF_Predictions = rf_predictions,
  SVM_Predictions = svm_predictions,
  Logistic_Predictions = multinom_predictions,
  DT_Predictions = dt_predictions
)

# Plot to compare predictions
ggplot(results_df) +
  geom_bar(aes(x = Actual, fill = "Actual"), alpha = 0.5, position = "dodge") +
  geom_bar(aes(x = RF_Predictions, fill = "RF Predictions"), alpha = 0.5, position = "dodge") +
  labs(title = "Actual vs Random Forest Predictions", x = "Species", y = "Count") +
  theme_minimal()

# Probability predictions for Random Forest
rf_prob_predictions <- predict(rf_model, test_data, type = "prob")
head(rf_prob_predictions)

# ROC Curve
multiclass_roc <- multiclass.roc(test_data$Species, rf_prob_predictions)
print(multiclass_roc)

# New prediction for a custom input
new_iris <- data.frame(
  Sepal.Length = 5.1,
  Sepal.Width = 3.5,
  Petal.Length = 1.4,
  Petal.Width = 0.2
)
custom_prediction <- predict(rf_model, new_iris)
print("Prediction for custom input:")
print(custom_prediction)
