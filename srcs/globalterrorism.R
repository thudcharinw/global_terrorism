getwd()
#devtools::install_github('topepo/caret/pkg/caret', build_vignettes = FALSE) 

#Change your working directory
setwd("/Users/ThudcharinW/Documents/MIS620/Project/global_terrorism/srcs")

library("rpart")
library("rpart.plot")
library("dplyr")
library("e1071")
library("caret")
library("ROCR")
library("plyr")
library("dplyr")
library("doParallel")
library("DMwR")
library("ROSE")
library("pROC")

#Download the dataset and place the file into working directory
#sample <- read.csv("globalterrorismdb_0617dist.csv",header=TRUE,sep=",")
#saveRDS(sample, "globalterrorism.rds")
sample <- readRDS("globalterrorism.rds")

#glimpse(sample)
#str(sample)
#head(sample)
#names(sample)
sample <- sample[,c("iyear","imonth","iday",
                    "country","region",
                    "extended", "crit1", "crit2", "crit3", "doubtterr", "multiple", "suicide", "ishostkid", "ransom", "INT_ANY",
                    "attacktype1","weaptype1","weapsubtype1",
                    "targtype1","natlty1","propextent",
                    "nperps","nkill","nwound",
                    "gname","success")] 
str(sample)

# analyze data in 1998 and after to avoid non-recorded data.
sample <- sample[sample$iyear >= 1998, ]

sample$iyear <- as.factor(sample$iyear)
sample$imonth <- as.factor(sample$imonth)
sample$iday <- as.factor(sample$iday)
sample$country <- as.factor(sample$country)
sample$region <- as.factor(sample$region)
sample$attacktype1 <- as.factor(sample$attacktype1)
sample$weaptype1 <- as.factor(sample$weaptype1)
sample$weapsubtype1 <- as.factor(sample$weapsubtype1) #NA
sample$targtype1 <- as.factor(sample$targtype1)
sample$natlty1 <- as.factor(sample$natlty1) #NA
sample$propextent <- as.factor(sample$propextent) #NA
sample$success <- as.factor(sample$success)
sample$extended <- as.factor(sample$extended)
sample$crit1 <- as.factor(sample$crit1)
sample$crit2 <- as.factor(sample$crit2)
sample$crit3 <- as.factor(sample$crit3)
sample$doubtterr <- as.factor(sample$doubtterr)
sample$multiple <- as.factor(sample$multiple)
sample$suicide <- as.factor(sample$suicide)
sample$ishostkid <- as.factor(sample$ishostkid) #NA
sample$ransom <- as.factor(sample$ransom) #NA
sample$INT_ANY <- as.factor(sample$INT_ANY)
#which(is.na(sample$doubtterr))
sample$gname <- as.character(sample$gname)

# replace NA Value
sample$propextent[which(is.na(sample$propextent))] <- 4 #replace NA with 4 = Unknown
levels(sample$natlty1) <- c(levels(sample$natlty1),"-99")
sample$natlty1[which(is.na(sample$natlty1))] <- -99 #replace NA with -99
levels(sample$weapsubtype1) <- c(levels(sample$weapsubtype1),"-99")
sample$weapsubtype1[which(is.na(sample$weapsubtype1))] <- -99 #replace NA with -99
sample$ishostkid[which(is.na(sample$ishostkid))] <- -9 #replace NA with -9 = Unknown
sample$ransom[which(is.na(sample$ransom))] <- -9 #replace NA with -9 = Unknown

summary(sample$nkill)
sample$nkill[which(is.na(sample$nkill))] <- 1 # median = 1 assigned to NA

summary(sample$nwound)
sample$nwound[which(is.na(sample$nwound))] <- 0 # median = 0 assigned to NA
#summary(sample$weapsubtype1)
#table(sample$natlty1)
#which(is.na(sample$natlty1))

#sample <- subset(sample, gname != 'Unknown')
#sample <- sample[sample$country_txt == 'United States',]
#summary(sample) 

########## Predict success ##########

levels(sample$success)[levels(sample$success)=="1"] <- "SUCCESS"
levels(sample$success)[levels(sample$success)=="0"] <- "FAIL"
drops <- c("country", "nperps", "gname", "natlty1")
sample <- sample[,!(names(sample) %in% drops)]
table(sample$success)

set.seed(1000)
inTrain <- createDataPartition(y=sample$success, p=.80, list=FALSE)
trainSplit <- sample[inTrain,]
length(trainSplit$success)
testSplit <- sample[-inTrain,]
length(testSplit$success)

#cl <- makeCluster(detectCores())
#registerDoParallel(cl)
#getDoParWorkers()
#set.seed(1000)
#smote_train <- SMOTE(success ~ ., data  = trainSplit)   
#stopCluster(cl)
#table(smote_train$success)

cl <- makeCluster(detectCores())
registerDoParallel(cl)
getDoParWorkers()
set.seed(1000)
rose_train <- ROSE(success ~ ., data  = trainSplit)$data
stopCluster(cl)
table(rose_train$success)

#set.seed(1000)
#?downSample
#down_train <- downSample(x = trainSplit[,!(names(trainSplit) %in% c("success"))],
#                         y = trainSplit$success, yname = "success")
#table(down_train$success)

#set.seed(1000)
#up_train <- upSample(x = trainSplit[,!(names(trainSplit) %in% c("success"))],
#                         y = trainSplit$success, yname = "success")
#table(up_train$success)

y.train <- trainSplit$success
x.train <- trainSplit[,!(names(trainSplit) %in% c("success"))]

#y.train.up <- up_train$success
#x.train.up <- up_train[,!(names(up_train) %in% c("success"))]

#y.train.down <- down_train$success
#x.train.down <- down_train[,!(names(down_train) %in% c("success"))]

#y.train.smote <- smote_train$success
#x.train.smote <- smote_train[,!(names(smote_train) %in% c("success"))]

y.train.rose <- rose_train$success
x.train.rose <- rose_train[,!(names(rose_train) %in% c("success"))]

y.test <- testSplit$success
x.test <- testSplit[,!(names(testSplit) %in% c("success"))]

ctrl <- trainControl(method="cv", number=5,
                     classProbs=TRUE,
                     summaryFunction = twoClassSummary, # twoClassSummary for binary
                     allowParallel =  TRUE)

##### NaiveBayes #####

#run model in parallel
cl <- makeCluster(detectCores())
registerDoParallel(cl)

getDoParWorkers()

set.seed(100)
m.nb.rose <- train(y=y.train.rose, x=x.train.rose,
              trControl = ctrl,
              metric = "ROC",
              method = "nb")

stopCluster(cl)
#saveRDS(m.nb, "m_nb.rds")
getTrainPerf(m.nb.rose)
varImp(m.nb.rose)

#pred <- prediction(p.nb, y.test)
#perf <- performance(pred, "tpr", "fpr")

##### Decision Tree #####

cl <- makeCluster(detectCores())
registerDoParallel(cl)

getDoParWorkers()

set.seed(100)
m.rpart.rose <- train(y=y.train.rose, x=x.train.rose,
              trControl = ctrl,
              metric = "ROC",
              method = "rpart")
stopCluster(cl)
m.rpart.rose
#saveRDS(m.rpart, "m_rpart.rds")
getTrainPerf(m.rpart.rose)
varImp(m.rpart.rose)

##### Random Forest ##### xxx cannot handle > 53 categories /// yes

cl <- makeCluster(detectCores())
registerDoParallel(cl)

getDoParWorkers()

set.seed(100)
m.rf.rose <- train(y=y.train.rose, x=x.train.rose,
                 trControl = ctrl,
                 metric = "ROC",
                 method = "rf")
              #tuneGrid = grid) #mtry=11 best ROC

stopCluster(cl)

m.rf.rose
#saveRDS(m.rf, "m_rf.rds")
getTrainPerf(m.rf.rose)
varImp(m.rf.rose)

p.rf <- predict(m.rf,x.test)
confusionMatrix(p.rf,y.test)

##### Tree Bag #####

cl <- makeCluster(detectCores())
registerDoParallel(cl)

getDoParWorkers()

set.seed(100)
m.treebag.rose <- train(y=y.train.rose, x=x.train.rose,
              trControl = ctrl,
              metric = "ROC",
              method = "treebag")

stopCluster(cl)

m.treebag.rose
#saveRDS(m.treebag, "m_treebag.rds")
getTrainPerf(m.treebag.rose)
varImp(m.treebag.rose)

##### Neural Network #####

cl <- makeCluster(detectCores())
registerDoParallel(cl)

getDoParWorkers()

set.seed(100)
m.nnet.rose <- train(y=y.train.rose, x=x.train.rose,
                        trControl = ctrl,
                        metric = "ROC",
                        method = "nnet")

stopCluster(cl)

m.nnet.rose
#saveRDS(m.treebag, "m_treebag.rds")
getTrainPerf(m.nnet.rose)
varImp(m.nnet.rose)

##### Boosting ##### xxx

cl <- makeCluster(detectCores())
registerDoParallel(cl)

getDoParWorkers()

set.seed(100)
m.ada.rose <- train(y=y.train.rose, x=x.train.rose,
                   trControl = ctrl,
                   metric = "ROC",
                   method = "ada")

stopCluster(cl)
m.ada.rose
getTrainPerf(m.ada.rose)
varImp(m.ada.rose)

#######################
##### PERFORMANCE #####
#######################

# Compare TRAINING performance of cross-validation runs
rValues <- resamples(list(naiveBayes=m.nb.rose, rpart=m.rpart.rose, randomForest=m.rf.rose, treebag=m.treebag.rose, ada=m.ada.rose, nnet=m.nnet.rose))

bwplot(rValues, metric="ROC")
bwplot(rValues, metric="Sens") #Sensitvity
bwplot(rValues, metric="Spec")

p.treebag <- predict(m.treebag.rose, x.test)
confusionMatrix(p.treebag, y.test) #0.7341

p.nnet <- predict(m.nnet.rose, x.test)
confusionMatrix(p.nnet, y.test) #0.6964

p.rpart <- predict(m.rpart.rose, x.test)
confusionMatrix(p.rpart, y.test) #0.3632

p.rf <- predict(m.rf.rose, x.test)
confusionMatrix(p.rf, y.test) #0.8149

p.nb <- predict(m.nb.rose, x.test)
confusionMatrix(p.nb, y.test) #0.7865

p.ada <- predict(m.ada.rose, x.test)
confusionMatrix(p.ada, y.test) #0.4857

p.treebag.prob <- predict(m.treebag.rose, x.test, type = "prob")
p.nnet.prob <- predict(m.nnet.rose, x.test, type = "prob")
p.rf.prob <- predict(m.rf.rose, x.test, type = "prob")
p.nb.prob <- predict(m.nb.rose, x.test, type = "prob")

# using FAIL as positive class
p.treebag.roc <- roc(response = y.test, predictor = p.treebag.prob$FAIL)
p.nnet.roc <- roc(response = y.test, predictor = p.nnet.prob$FAIL)
p.rf.roc <- roc(response = y.test, predictor = p.rf.prob$FAIL)
p.nb.roc <- roc(response = y.test, predictor = p.nb.prob$FAIL)
auc(p.treebag.roc)
auc(p.nnet.roc)
auc(p.rf.roc)
auc(p.nb.roc)

plot(p.treebag.roc, col="black")
plot(p.nnet.roc, add=T, col="red")
plot(p.nb.roc, add=T, col="blue")
legend(x=.34, y=.3, cex=1, legend=c("treebag","nnet", "naiveBayes"), col=c("black", "red", "blue"), lwd=5)

varImp(m.treebag.rose)
