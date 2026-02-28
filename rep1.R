options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install packages (if not already installed)
install.packages(c("tidyverse", "ggplot2", "dplyr", "tidyr", "corrplot", "lubridate", "summarytools"))

# Load libraries
library(tidyverse)
library(ggplot2)
library(dplyr)
library(summarytools)  # For descriptive statistics
library(MASS)
library(caret)
library(rpart)
library(rpart.plot)
library(cluster)    # For clustering algorithms
library(factoextra) # For visualization
library(dendextend) # For dendrograms
library(gridExtra)
library(klaR)


file.choose()
caschools = read.csv("C:\\Users\\HP\\Downloads\\CASchools.csv")
head(caschools)
dim(caschools)

#Chapter 3 Data Cleaning

# Verify structure
str(caschools)
summary(caschools)

# Convert categoricals to factors (optional)
caschools$county <- as.factor(caschools$county)
caschools$grades <- as.factor(caschools$grades)

# Handle hypothetical missing values (though your dataset likely has none)
sum(is.na(caschools))  # Confirm no NAs exist  
caschools <- na.omit(caschools)  # Only if NAs exist

#outliers detection
boxplot(caschools$math, caschools$read)  # Visual check  
  caschools[which(caschools$students > 5000), ]  # Example: Large districts  

# Create meaningful new variables
caschools$student_teacher_ratio <- caschools$students / caschools$teachers
caschools$spending_per_student <- caschools$expenditure / caschools$students
# Create binary math performance groups
caschools$math_group <- ifelse(caschools$math > median(caschools$math), "High", "Low")
caschools$math_group <- as.factor(caschools$math_group)

# Verify creation
head(caschools[c("math", "math_group")])
# Scale ONLY for KNN, clustering, LDA/QDA
scale_cols <- c("income", "expenditure", "students", "teachers", "math", "read")
caschools_scaled <- as.data.frame(scale(caschools[scale_cols]))

# Keep categoricals and new features
caschools_scaled <- cbind(caschools_scaled, 
                          caschools[!names(caschools) %in% scale_cols])

set.seed(123)
train_idx <- sample(1:nrow(caschools), 0.8 * nrow(caschools))
train <- caschools[train_idx, ]
test <- caschools[-train_idx, ]

#CHapter 4 Data Exploration

summary(caschools$math)
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#   605.0   640.0   654.0   654.2   666.0   706.0
# Histogram

ggplot(caschools, aes(x = math)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Math Scores")

# Boxplot
ggplot(caschools, aes(y = math)) +
  geom_boxplot(fill = "orange") +
  labs(title = "Boxplot of Math Scores")

#read_scr
summary(caschools$read)
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#   604.0   640.0   655.0   655.7   671.0   704.0

#visualization
# Histogram
ggplot(caschools, aes(x = read)) +
  geom_histogram(bins = 30, fill = "lightgreen", color = "black") +
  labs(title = "Distribution of Reading Scores")

summary(caschools$income)
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#   5.335   8.978   9.824  13.147  10.415  22.690

#Visualization
ggplot(caschools, aes(x = income)) +
  geom_histogram(bins = 20, fill = "gold", color = "black") +
  labs(title = "Distribution of Income")

summary(caschools$english)
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#   0.000   0.000   4.584  15.768  13.858  30.000

#Visualization
ggplot(caschools, aes(x = english)) +
  geom_histogram(bins = 20, fill = "salmon", color = "black") +
  labs(title = "Distribution of English Learners (%)")




table(caschools$county)
#   Alameda       Butte    Fresno  LosAngeles    Mendocino      Orange 
#        18           6          8         73           4          20 

ggplot(caschools, aes(x = county)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Schools by County") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

table(caschools$grades)
# KK-08 
#     6


#MULTIVARIATE
ggplot(caschools, aes(x = income, y = math, color = county)) +
  geom_point(size = 3) +
  labs(title = "Math Scores vs. Income by County") +
  theme_minimal()

ggplot(caschools, aes(x = english, y = math)) +
  geom_point(aes(color = county)) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "English Learners (%) vs. Math Scores")

caschools <- caschools %>% mutate(expenditure_per_student = expenditure / students)

ggplot(caschools, aes(x = expenditure_per_student, y = math)) +
  geom_point(aes(color = county)) +
  labs(title = "Expenditure per Student vs. Math Scores")


#Regression
# Plot and model for all schools
#1
plot(math ~ income, pch = 21, col = "blue", data = caschools,
     xlab = "Income (thousands of USD)", ylab = "Math Scores")
model_all <- lm(math ~ income, data = caschools)
abline(model_all, lwd = 2, col = "green")

# Model summary
summary(model_all)

#2
plot(math ~ english, pch = 21, col = "purple", data = caschools,
     xlab = "English Learners (%)", ylab = "Math Scores")
model_english <- lm(math ~ english, data = caschools)
abline(model_english, lwd = 2, col = "red")

summary(model_english)

#3
combined_model <- lm(math ~ income + english, data = caschools)
summary(combined_model)

#4
predict(combined_model, data.frame(income = 8.978, english = 30))
# Output: 634.5
predict(combined_model, data.frame(income = 22.69, english = 0))
# Output: 673.1



#CHapter 5 Attribute Selection 
attach(caschools)

# Full model with all candidate predictors
full_model <- lm(math ~ income + english + students + teachers + calworks + 
                   lunch + computer + expenditure + county + grades, 
                 data = caschools)

step_model <- stepAIC(full_model, direction = "both")
summary(step_model)


#Chapter 6 kNN Algorithm
knn_model = lm(math ~ income + english + lunch + calworks + computer + grades)
knn_model
summary(knn_model)




#normalization
normalize <- function(x) { return((x - min(x)) / (max(x) - min(x))) }
caschools_norm <- as.data.frame(lapply(caschools[, c("income", "english", "lunch", "calworks", "computer")], normalize))
dim(caschools_norm)
summary(caschools_norm)



#train-Test split
set.seed(123)
train_indices <- sample(1:nrow(caschools), 336)
train_data <- caschools_norm[train_indices, ]
test_data <- caschools_norm[-train_indices, ]
train_labels <- caschools$math[train_indices]
test_labels <- caschools$math[-train_indices]

#knn regression
knn_model <- knnreg(train_data, train_labels, k = 10)

predictions <- predict(knn_model, test_data)
library(caret)
postResample(pred = predictions, obs = test_labels)


#Visualization
ggplot(data.frame(Actual = test_labels, Predicted = predictions), 
       aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_abline(slope = 1, linetype = "dashed") +
  labs(title = "k-NN (k=10) Predictions vs. Actual Math Scores")


#Chapter 7 tree
# Regression Tree to predict math scores
tree_model <- rpart(math ~ income + english + lunch + calworks + computer + teachers+ computer+expenditure,
                    data = train, method = "anova")

# Visualize the tree
rpart.plot(tree_model, main = "Regression Tree for Math Scores")
tree_model2 <- rpart(english ~ income + math + lunch + calworks + computer + teachers+ computer+expenditure,
                    data = train, method = "anova")

# Visualize the tree
rpart.plot(tree_model2, main = "Regression Tree for English Scores")

#predictions math score 
# Predict on test data
tree_predictions <- predict(tree_model, newdata = test)

# Evaluate performance
postResample(pred = tree_predictions, obs = test$math)

#predictions english score 
# Predict on test data
tree_predictions2 <- predict(tree_model2, newdata = test)

# Evaluate performance
postResample(pred = tree_predictions2, obs = test$english)


#plotting actual vs pred
#math scr
ggplot(data.frame(Actual = test$math, Predicted = tree_predictions), 
       aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_abline(slope = 1, linetype = "dashed") +
  labs(title = "Decision Tree Predictions vs. Actual Math Scores")

#english scr
ggplot(data.frame(Actual = test$english, Predicted = tree_predictions2), 
       aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "brown") +
  geom_abline(slope = 1, linetype = "dashed") +
  labs(title = "Decision Tree Predictions vs. Actual English Scores")


#Chapter 8 Cluster Analysis

features <- c("income", "english", "lunch", "calworks", "computer")
scaled_data <- scale(caschools[, features])  # Standardize for clustering

#kmeans clustering 
fviz_nbclust(scaled_data, kmeans, method = "wss") + 
  geom_vline(xintercept = 3, linetype = 2)  # Elbow at k=3
set.seed(123)
kmeans_result <- kmeans(scaled_data, centers = 3, nstart = 25)

#Visualization
fviz_cluster(kmeans_result, data = scaled_data, geom = "point")

#hierarchial CLustering
dist_matrix <- dist(scaled_data, method = "euclidean")
hc <- hclust(dist_matrix, method = "ward.D2")

# Cut tree into 3 clusters
hc_clusters <- cutree(hc, k = 3)

#dendogram


# Step 1: Aggregate data by similar schools first
agg_data <- aggregate(caschools[, c("income", "english", "math", "read")], 
                      by = list(cluster = cutree(hc, k = 20)),  # Pre-group into 20 clusters
                      FUN = mean)

# Step 2: Create dendrogram with enhanced labels
hc_agg <- hclust(dist(scale(agg_data[, -1])), method = "ward.D2")

# Create meaningful labels showing key stats
agg_data$label <- paste(
  "Cluster ", agg_data$cluster, "\n",
  "Income: $", round(agg_data$income, 1), "k\n",
  "English: ", round(agg_data$english, 1), "%\n",
  "Math: ", round(agg_data$math, 0), 
  sep = ""
)

dend_agg <- as.dendrogram(hc_agg) %>%
  set("branches_k_color", k = 3) %>% 
  set("labels", agg_data$label) %>%
  set("labels_cex", 0.7) %>%
  set("labels_col", "darkblue") %>%
  set("branches_lwd", 1.5)

# Step 3: Plot with annotations
par(mar = c(5, 2, 1, 15))  # Increase right margin for labels
plot(dend_agg, horiz = TRUE, 
     main = "Hierarchical Clustering of California Schools\n(Grouped by Income, English Learners, and Test Scores)",
     xlab = "Dissimilarity (Ward's Method)")

# Add cluster interpretation
legend("topright", 
       legend = c("High-Income, High Scores", 
                  "Mid-Range", 
                  "Low-Income, High English Learners"),
       fill = c("#1b9e77", "#d95f02", "#7570b3"),  # Match dendrogram colors
       border = NA,
       bty = "n",
       cex = 0.8)

# Add reference lines
abline(v = c(2, 4), lty = 2, col = "gray50")


#Chapter 9 LDA & QDA
# Create binary math performance groups
caschools$math_group <- ifelse(caschools$math > median(caschools$math), "High", "Low")
caschools$math_group <- as.factor(caschools$math_group)

set.seed(123)
train_idx <- sample(1:nrow(caschools), 0.8 * nrow(caschools))
train <- caschools[train_idx, ]
test <- caschools[-train_idx, ]
#LDA 
lda_model <- lda(math_group ~ income + english + expenditure_per_student, data = train)
 lda_pred <- predict(lda_model, test)
confusionMatrix(lda_pred$class, test$math_group)

#plot LDA

ggplot(test, aes(x = income, y = english, color = lda_pred$class)) +
  geom_point(size = 3) +
  geom_vline(xintercept = mean(caschools$income), linetype = "dashed") +
  labs(title = "LDA Classification: Income vs. English Learners",
       x = "Income", y = "English Learners (%)")


#QDA
qda_model <- qda(math_group ~ income + english + expenditure_per_student, data = train)
qda_pred <- predict(qda_model, test)
confusionMatrix(qda_pred$class, test$math_group)

plot1 <- ggplot(test, aes(x = income, color = lda_pred$class)) + geom_density() + ggtitle("LDA")
plot2 <- ggplot(test, aes(x = income, color = qda_pred$class)) + geom_density() + ggtitle("QDA")
grid.arrange(plot1, plot2, ncol = 2)

# For LDA
partimat(math_group ~ income + english + expenditure_per_student, 
         data = train, 
         method = "lda",
         main = "LDA Classification Boundaries",
         col.correct = "forestgreen",  # Correct classifications
         col.wrong = "firebrick")      # Misclassifications

# For QDA
partimat(math_group ~ income + english + expenditure_per_student, 
         data = train, 
         method = "qda",
         main = "QDA Classification Boundaries",
         col.correct = "forestgreen",
         col.wrong = "firebrick")

