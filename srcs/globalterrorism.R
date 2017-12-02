getwd()

#Change your working directory
setwd("/Users/ThudcharinW/Documents/MIS620/Project/global_terrorism/srcs")

library("rpart")
library("rpart.plot")
library("dplyr")
library("e1071")
library("caret")
library("ROCR")
library("MLmetrics")

#Download the dataset and place the file into working directory
#sample <- read.csv("globalterrorismdb_0617dist.csv",header=TRUE,sep=",")
#saveRDS(sample, "globalterrorism.rds")
sample <- readRDS("globalterrorism.rds")

#sample1 <- select(sample, country_txt, gname)
#head(sample1)

glimpse(sample)
str(sample)
head(sample)
names(sample)
#sample <- sample[,c("iyear","imonth","iday",
#                    "country_txt","region_txt",
#                    "attacktype1_txt","weaptype1_txt","weapsubtype1_txt",
#                    "targtype1_txt","natlty1_txt","nperps","nkill",
#                    "nwound","success","gname")] #propvalue, INT_ANY, scite1, alternative, latitude, longitude, doubtterr, multiple
sample <- sample[,c("iyear","imonth","iday",
                    "country","region",
                    "attacktype1","weaptype1","weapsubtype1",
                    "targtype1","natlty1","nperps","nkill",
                    "nwound","success","gname")]

str(sample)

sample$imonth <- as.factor(sample$imonth)
sample$iday <- as.factor(sample$iday)
sample$country <- as.factor(sample$country)
sample$region <- as.factor(sample$region)
sample$attacktype1 <- as.factor(sample$attacktype1)
sample$weaptype1 <- as.factor(sample$weaptype1)
sample$weapsubtype1 <- as.factor(sample$weapsubtype1)
sample$targtype1 <- as.factor(sample$targtype1)
sample$natlty1 <- as.factor(sample$natlty1)

sample$gname <- as.character(sample$gname)

#in case we want to analyze data in 1997 and after to avoid missing data.
sample <- sample[sample$iyear >= 1997, ]
sample$iyear <- as.factor(sample$iyear)

#sample <- subset(sample, gname != 'Unknown')
#sample <- sample[sample$country_txt == 'United States',]
#nrow(sample)
#summary(sample) 

#subset only the records of ISIL
isil <- subset(sample, grepl("ISIL", sample$gname))
isil$gname <- "ISIL"
#table(isil$gname)
#table(isil$success)

#subset only the records of Al-Qaida
alqaida <- subset(sample, grepl("Al-Qaida", sample$gname)) #subset(sample, sample$gname == 'Al-Qaida')
alqaida$gname <- "AQD"
#table(alqaida$gname)
#table(alqaida$success)
#str(alqaida)
#unique(alqaida$gname)

#subset only the records of Taliban
taliban <- subset(sample, sample$gname == 'Taliban')
#str(taliban)
#table(taliban$gname)
#table(taliban$success)

data <- rbind(taliban, isil, alqaida)
data$gname <- as.factor(data$gname)
data$success <- as.factor(data$success)
#levels(data$gname)[levels(data$gname)=="Islamic State of Iraq and the Levant (ISIL)"] <- "ISIL"
str(data)

y <- data$gname
x <- data[,c(1:10,14)]
table(data$gname)

set.seed(100)
inTrain <- createDataPartition(y=y, p=.85, list=FALSE)
y.train <- y[inTrain]
x.train <- x[inTrain,]
#which(is.na(x.train$nkill))

y.test <- y[-inTrain]
x.test <- x[-inTrain,]
levels(y.test)

ctrl <- trainControl(method="cv", number=5,
                     classProbs=TRUE,
                     #function used to measure performance
                     summaryFunction = multiClassSummary, #for non binary, twoClassSummary for binary
                     allowParallel =  FALSE)

##### NaiveBayes #####

set.seed(100)
m.nb <- train(y=y.train, x=x.train,
              trControl = ctrl,
              metric = "ROC", #using AUC to find best performing parameters
              method = "nb")
m.nb
getTrainPerf(m.nb)
varImp(m.nb)
plot(m.nb)
p.nb<- predict(m.nb,x.test)
confusionMatrix(p.nb,y.test)
nb.prob <- predict(m.nb, x.test, type="prob")
nb.prob

##### rpart #####

set.seed(100)
m.rpart <- train(y=y.train, x=x.train,
                 trControl = ctrl,
                 metric = "ROC", #using AUC to find best performing parameters
                 method = "rpart") #change method as you want to use different models
m.rpart
getTrainPerf(m.rpart)

##### Neural Network #####

set.seed(100)
m.nnet<- train(y=y.train, x=x.train,
              trControl = ctrl,
              metric = "Accuracy",
              method = "nnet")
m.nnet
getTrainPerf(m.nnet)

m.nnet<- predict(m.nnet,x.test)
confusionMatrix(m.nnet,y.test)

###########################




#set.seed(100)
#?rpart
#fit <- rpart(success ~ ., 
#             method="class", 
#             data=taliban,
#             control=rpart.control(minsplit=1, maxdepth = 3), # the result sholud have at least minimum one split
#             parms=list(split='information'))
#class(fit)
#names(fit)
#summary(fit)
#?rpart.plot
#rpart.plot(fit, type=0, extra=2, clip.right.labs=FALSE, varlen=0, faclen=0)
#rpart.plot(fit)


#train_sample <- sample[1:300,]
#test_sample <- sample[301:364,]
#drops <- c("gname")
#test_sample2 <- test_sample[,!(names(test_sample) %in% drops)]
#actual_gname <- test_sample$gname
#model <- naiveBayes(gname ~ ., train_sample)
#model
#saveRDS(model, "naiveBayesModel.rds")
#result <- predict(model, test_sample2, type = "raw")


#result
#test_sample$gname[1:64]
#table(result == actual_gname)

#pred <- prediction(result, actual_gname)
#perf <- performance(pred, "tpr", "fpr")

#write.csv(result, file = "result.csv")
#write.csv(actual_gname, file = "actual.csv")
#result
#actual_gname

#confusionMatrix(result, test_sample$gname[555:666])
#pred <- prediction(result, test_sample$gname[555:666])
#head(x)
#which(is.na(x$natlty1_txt))
