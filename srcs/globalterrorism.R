getwd()

#Change your working directory
setwd("/Users/ThudcharinW/Documents/MIS620/Project/global_terrorism/srcs")

library("rpart")
library("rpart.plot")
library("dplyr")
library("e1071")
library("caret")
library("ROCR")

#Download the dataset and place the file into working directory
#sample <- read.csv("globalterrorismdb_0617dist.csv",header=TRUE,sep=",")
#saveRDS(sample, "globalterrorism.rds")
sample <- readRDS("globalterrorism.rds")

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
                    "targtype1","natlty1","propextent","success",
                    "nperps","nkill","nwound",
                    "gname")]

str(sample)

# analyze data in 1997 and after to avoid non-recorded data.
sample <- sample[sample$iyear >= 1997, ]

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

sample$gname <- as.character(sample$gname)

# replace NA Value
sample$propextent[which(is.na(sample$propextent))] <- 4 #replace NA with 4 = Unknown
levels(sample$natlty1) <- c(levels(sample$natlty1),"-99")
sample$natlty1[which(is.na(sample$natlty1))] <- -99 #replace NA with -99
levels(sample$weapsubtype1) <- c(levels(sample$weapsubtype1),"-99")
sample$weapsubtype1[which(is.na(sample$weapsubtype1))] <- -99 #replace NA with -99
#summary(sample$weapsubtype1)
#table(sample$natlty1)
which(is.na(sample$natlty1))

#sample <- subset(sample, gname != 'Unknown')
#sample <- sample[sample$country_txt == 'United States',]
#summary(sample) 

##### determince nkill cluster #####

#searching for optimal K
summary(sample$nkill)
sample$nkill[which(is.na(sample$nkill))] <- 1 # median = 1 assigned to NA

#rescale
#sample$nkill <- sample$nkill/sd(sample$nkill)

#wss <- numeric(15)
#for (k in 1:15) 
#{
#  clust <- kmeans(sample$nkill, centers=k, nstart=25, algorithm = "Lloyd")
#  
#  wss[k] <- sum(clust$withinss)
#}
#plot(1:15, wss, type="b", xlab="Number of Clusters", ylab="Within Sum of Squares") 

set.seed(1000)
km = kmeans(sample$nkill, 5, nstart=25, algorithm = "Lloyd")
km$cluster
km$size
km$centers
sample$nkill.cat <- km$cluster
sample$nkill.cat <- as.factor(sample$nkill.cat)
table(sample$nkill.cat)

##### determince nwound cluster #####

summary(sample$nwound)
sample$nwound[which(is.na(sample$nwound))] <- 0 # median = 0 assigned to NA

#rescale
#sample$nwound <- sample$nwound/sd(sample$nwound)

#wss <- numeric(15)
#for (k in 1:15) 
#{
#  clust <- kmeans(sample$nwound, centers=k, nstart=25, algorithm = c("Lloyd"))
#  
#  wss[k] <- sum(clust$withinss)
#}
#plot(1:15, wss, type="b", xlab="Number of Clusters", ylab="Within Sum of Squares") 

set.seed(2017)
km = kmeans(sample$nwound, 4, nstart=25, algorithm = c("Lloyd"))
which(km$cluster==1)
km$size
km$centers
sample$nwound.cat <- km$cluster
sample$nwound.cat <- as.factor(sample$nwound.cat)
table(sample$nwound.cat)

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

# combine data of Taliban, ISIL, Al-Qaida
data <- rbind(taliban, isil, alqaida)
data$gname <- as.factor(data$gname)
#levels(data$gname)[levels(data$gname)=="Islamic State of Iraq and the Levant (ISIL)"] <- "ISIL"
str(data)

########## Predict gname ##########

y <- data$gname
x <- data[,c(1:12, 17:18)]
table(data$gname)

set.seed(100)
inTrain <- createDataPartition(y=y, p=.75, list=FALSE)
y.train <- y[inTrain]
x.train <- x[inTrain,]

y.test <- y[-inTrain]
x.test <- x[-inTrain,]

ctrl <- trainControl(method="cv", number=5,
                     classProbs=TRUE,
                     summaryFunction = multiClassSummary, #for non binary, twoClassSummary for binary
                     allowParallel =  FALSE)

##### NaiveBayes #####

modelLookup("nb")
set.seed(100)
m.nb <- train(y=y.train, x=x.train,
              trControl = ctrl,
              metric = "Accuracy", #"ROC" for binary classification
              method = "nb",
              tuneGrid = data.frame(fL=TRUE, usekernel=FALSE, adjust=FALSE))
m.nb
getTrainPerf(m.nb)
varImp(m.nb)

p.nb<- predict(m.nb,x.test)
table(p.nb, y.test)
confusionMatrix(p.nb,y.test)

plot(m.nb)

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

modelLookup("nnet")
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

########## Predict success ##########

##### NaiveBayes #####
y <- sample$success
levels(y)[levels(y)=="1"] <- "SUCCESS"
levels(y)[levels(y)=="0"] <- "FAIL"
x <- sample[,c(1:11, 16:18)]
x$gname <- as.factor(x$gname)
x$gname <- as.numeric(x$gname)
x$gname <- as.factor(x$gname)

set.seed(100)
inTrain <- createDataPartition(y=y, p=.75, list=FALSE)
y.train <- y[inTrain]
x.train <- x[inTrain,]

y.test <- y[-inTrain]
x.test <- x[-inTrain,]

ctrl <- trainControl(method="cv", number=5,
                     classProbs=TRUE,
                     summaryFunction = twoClassSummary, # twoClassSummary for binary
                     allowParallel =  FALSE)
set.seed(100)
m.nb <- train(y=y.train, x=x.train,
              trControl = ctrl,
              metric = "ROC",
              method = "nb")
m.nb
getTrainPerf(m.nb)
varImp(m.nb)

p.nb<- predict(m.nb,x.test)
table(p.nb, y.test)
confusionMatrix(p.nb,y.test)

##### Neural Network #####


#########################################