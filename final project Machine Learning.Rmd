## My Practical Machine Learning Course Project
#By Diana López, Date: "03rd Jan 2019"

## Description
#One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
#In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

#The training data for this project are available here:

# [Training set](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
# [Test set](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

#The objective is to correctly predict the variable `classe` of the `Test set`.
#This variable indicates how well the exercise is performed. 
#The value `A` indicates that the exercise was well performed while the other letters (from `B` to `E`) respectively indicate that common mistakes has been done during the execution of the weightlifting.

##The steps to realize the project are:
#1. loading the required packages
#2. loading the data
#3. cleaning the data
#4. building the models
#4.1. classification trees
#4.2. random forest
#4.3. general boosted regression
#5. Validating the best model

## Preparation

### Loading required packages

library(caret)
library(ggplot2)
library(randomForest)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
install.packages("rattle")
install.packages("gmb")
library(rattle)
#library(corrplot)
library(gbm)


### Loading the data


ml_path<-"~/R/Data Science Coursera/"
if(!file.exists(paste(ml_path,"pml-training.csv"))){
  fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(fileUrl,destfile="./pml-training.csv")
}
if(!file.exists(paste(ml_path,"pml-testing.csv"))){
  fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(fileUrl,destfile="./pml-testing.csv")
}



training_in <- read.csv("pml-training.csv")
test_in <- read.csv("pml-testing.csv")


### Data cleaning
#All the variables which contain all NA values are discarded.


trainData<- training_in[, colSums(is.na(training_in)) == 0]
validData <- test_in[, colSums(is.na(test_in)) == 0]

#Removing the first seven variables as they have little impact on the outcome classe.

trainData <- trainData[, -c(1:7)]
validData <- validData[, -c(1:7)]

##Modeling

## Preparing the datasets for prediction

#First the `training` dateset is splitted in two datasets:
  
# `trainData`: will be the dataset used to train and test the models
# `validData` : will be the dataset used to validate the models

set.seed(1234) 
inTrain <- createDataPartition(trainData$classe, p = 0.7, list = FALSE)
trainData <- trainData[inTrain, ]
testData <- trainData[-inTrain, ]

#Cleaning even further by removing the variables that are near-zero-variance
NZV <- nearZeroVar(trainData)
trainData <- trainData[, -NZV]
testData  <- testData[, -NZV]

#Models are first trained. Then they are used with the test dataset. 
#Finally a confusion matrix is produced which can be checked to assess the accuracy of the models applied on the validation dataset.

##Classification trees

#Obtaining the model
set.seed(12345)
decisionTreeMod <- rpart(classe ~ ., data=trainData, method="class")

#Using the fancyRpartPlot() function to plot the classification tree as a dendogram.
fancyRpartPlot(decisionTreeMod)

#Testing the model “decisionTreeMod” on the testData to find out how well it performs by looking at the accuracy variable.
predictTreeMod <- predict(decisionTreeMod, testData, type = "class")
cmtree <- confusionMatrix(predictTreeMod, testData$classe)
plot(cmtree$table, col = cmtree$byClass, 
     main = paste("Classification Tree Confusion Matrix: Accuracy =", round(cmtree$overall['Accuracy'], 3)))

##Random forest

#Obtaining the model
set.seed(12345)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modRF <- train(classe ~ ., data=trainData, method="rf", trControl=controlRF)
modRF$finalModel
plot(modRF)

#Testing the model obtained “modRF” on the test data to find out how well it performs by looking at the accuracy variable.
predictRF <- predict(modRF, newdata=testData)
cmrf <- confusionMatrix(predictRF, testData$classe)
plot(cmrf$table, col = cmrf$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(cmrf$overall['Accuracy'], 3)))

#Measuring variable importance
rffit <- randomForest(classe ~ ., data=testData, ntree=500, keep.forest=FALSE, importance=TRUE)
rffit$importance # relative importance of predictors (highest <-> most important)
varImpPlot(rffit) # plot results

##Generalized Boosted Regression 

#Obtaining the model
set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modGBM  <- train(classe ~ ., data=trainData, method = "gbm", trControl = controlGBM, verbose = FALSE)
modGBM$finalModel
print(modGBM)

#Testing the model obtained “modGBM” on the test data to find out how well it performs by looking at the accuracy variable.
predictGBM <- predict(modGBM, newdata=testData)
cmGBM <- confusionMatrix(predictGBM, testData$classe)
plot(cmGBM$table, col = cmGBM$byClass, main = paste("GBM Confusion Matrix: Accuracy =", round(cmGBM$overall['Accuracy'], 3)))


#The three models' parameters are:

#decisionTreeMod
#modRF
#modGBM


#The confusion matrix report:

#cmtree
#cmrf
#cmGBM

## Model selection
#It is possible to see that Random Forest produces the model with the highest accuracy, more than 99%.

##Apply the best model to the validation data
#By comparing the accuracy rate values of the three models,  the ‘Random Forest’ model is the winner. 

final_prediction <- predict(modRF, newdata=validData)
final_prediction