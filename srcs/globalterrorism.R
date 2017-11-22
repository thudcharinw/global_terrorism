getwd()

#Change your working directory
setwd("/Users/ThudcharinW/Documents/MIS620/Project/global_terrorism/srcs")

library("rpart")
library("rpart.plot")
library("dplyr")

#Download the dataset and place the file into working directory
sample <- read.csv("globalterrorismdb_0617dist.csv",header=TRUE,sep=",")
saveRDS(sample, "globalterrorism.rds")
sample <- readRDS("globalterrorism.rds")

sample1 <- select(sample, country_txt, gname)
head(sample1)

glimpse(sample)
str(sample)
head(sample)
names(sample)
sample <- sample[,c("iyear","imonth","iday","doubtterr","alternative",
                    "multiple","country_txt","region_txt","latitude","longitude",
                    "attacktype1_txt","weaptype1_txt","weapsubtype1_txt",
                    "targtype1_txt","natlty1_txt","gname","nperps","nkill",
                    "nwound","propvalue","INT_ANY","success")] #scite1
sample$gname <- as.character(sample$gname)

#in case we want to analyze data in 1997 and after to avoid missing data.
sample <- sample[sample$iyear >= 1997, ]
summary(sample)

#subset only the records of ISIL
isil <- subset(sample, grepl("ISIL", sample$gname))
table(isil$gname)

#subset only the records of Al-Qaida
alqaida <- subset(sample, sample$gname == 'Al-Qaida')
table(alqaida$gname)
head(alqaida)

#subset only the records of Taliban
taliban <- subset(sample, sample$gname == 'Taliban')
table(taliban$gname)

set.seed(100)
?rpart
fit <- rpart(success ~ ., 
             method="class", 
             data=taliban,
             control=rpart.control(minsplit=1, maxdepth = 3), # the result sholud have at least minimum one split
             parms=list(split='information'))
class(fit)
names(fit)
summary(fit)
?rpart.plot
rpart.plot(fit, type=0, extra=2, clip.right.labs=FALSE, varlen=0, faclen=0)
rpart.plot(fit)

