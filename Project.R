#read data
library(randomForest)
library(rpart)
library(mlbench)
library(dplyr)
library(ipred)
library(maboost)
library(class)
data <- read.csv("C:/Users/Biligiri Vasan/Desktop/Spring 2016/ML/train.csv",header = T,sep = ",",na.strings = c("?"))
data <- na.omit(data)
df <- data.frame(data)
datatrain <- data[]
classtrain <- data[2]
a <- as.vector(datatrain)
b <- as.vector(classtrain)
str(datatrain)
str(a)
#length(classtrain)
#classtrain[,1]
#head(datatrain[,])
#initial analysis
data_table <- data.frame(table(df$Category))
data_ordered <- data_table[order(data_table$Freq,decreasing = T),]
#data_ordered
data_table
#data_frame <- select(df,df$Dates,df$DayOfWeek,df$PdDistrict,df$Address,df$X,df$Y)
#data_frame <- df %>% select(df,df$Dates,df$DayOfWeek,df$PdDistrict,df$Address,df$X,df$Y)

#data.rf <- randomForest(as.formula(names(datatrain[1]),"~.",datatrain,classtrain))

#data.bagging <- bagging(df[,c(-2,-3,-6)], df$Category, mfinal=15,
 #                       control=rpart.control(maxdepth=5, minsplit=15))

data.boosting <- maboost(as.formula(paste(names(datatrain[1]),"~.")),datatrain,iter = 10)
#df$Dates + df$DayOfWeek + df$PdDistrict + df$Address + df$X + df$Y


iris
train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
knn(train, test, cl, k = 3, prob=TRUE)
attributes(.Last.value)
train
test
cl

index <- sample(1:nrow(data),round(0.5*nrow(data)))
train <- data[index,]
test <- data[-index,]
train_labels <- train[,2]
test_labels <- test[,2]
class_attr <- names(data[2])
col_names <- names(data[,-2])
knn_model <- knn(train[,-2],test[,-2],train_labels,k=9,prob=T,use.all = FALSE)
train
