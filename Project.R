#read data
library(randomForest)
library(rpart)
library(mlbench)
library(dplyr)
library(ipred)
library(maboost)
library(class)
library('chron')
library(adabag)

#Read Data
data <- read.csv("Dataset/train.csv",header = T,sep = ",")

#Data Formatting
#Avoid less frequent categories
category_data <- data.frame(table(data$Category))
category_ordered <- category_data[order(category_data$Freq,decreasing = T),]
category_needed <- head(category_ordered$Var1,n=22)
data_new <- data[which(data$Category %in% category_needed),]
data_new$Category <- factor(data_new$Category)


#Format Address Column
data_new$Address <- mapply(gsub, "^.*?Block of ","", data_new["Address"])

#Format date Column
data_new$Dates <- as.character(data_new$Dates)
dtparts = t(as.data.frame(strsplit(data_new$Dates,' ')))
thetimes = chron(dates=dtparts[,1],times=dtparts[,2],format=c('y-m-d','h:m:s'))

data_new$Year <- format(thetimes, format="%Y")
data_new$Month <- format(thetimes, format="%m")
data_new$Date <- format(thetimes, format="%d")
data_new$hour <- format(thetimes, format="%H")

data_new$Year <- as.factor(data_new$Year)
data_new$Month <- as.factor(data_new$Month)
data_new$Date <- as.factor(data_new$Date)
data_new$hour <- as.factor(data_new$hour)

data_formatted <- data_new[,c("Category","DayOfWeek","PdDistrict","Address","X","Y","Year","Month","Date","hour")]

sample_data<-sample(1:nrow(data_formatted),size = 0.8*nrow(data_formatted))
training_data_set<-data[sample_data,]
test_data_set<-data[-sample_data,]

form <- as.formula("Category ~ DayOfWeek+PdDistrict+X+Y+Year+Month+Date+hour")
bag_model <- bagging(form, training_data_set, mfinal = 100)




#data <- read.csv("C:/Users/Biligiri Vasan/Desktop/Spring 2016/ML/train.csv",header = T,sep = ",",na.strings = c("?"))
#data <- na.omit(data)
#df <- data.frame(data)
#datatrain <- data[]
#classtrain <- data[2]
#a <- as.vector(datatrain)
#b <- as.vector(classtrain)
#str(datatrain)
#str(a)
#length(classtrain)
#classtrain[,1]
#head(datatrain[,])
#initial analysis
#data_table <- data.frame(table(df$Category))
#data_ordered <- data_table[order(data_table$Freq,decreasing = T),]
#data_ordered
#data_table
#data_frame <- select(df,df$Dates,df$DayOfWeek,df$PdDistrict,df$Address,df$X,df$Y)
#data_frame <- df %>% select(df,df$Dates,df$DayOfWeek,df$PdDistrict,df$Address,df$X,df$Y)

#data.rf <- randomForest(as.formula(names(datatrain[1]),"~.",datatrain,classtrain))

#data.bagging <- bagging(df[,c(-2,-3,-6)], df$Category, mfinal=15,
#                       control=rpart.control(maxdepth=5, minsplit=15))

#data.boosting <- maboost(as.formula(paste(names(datatrain[1]),"~.")),datatrain,iter = 10)
#df$Dates + df$DayOfWeek + df$PdDistrict + df$Address + df$X + df$Y


#iris
#train <- rbind(iris3[1:25,,1], iris3[1:25,,2], iris3[1:25,,3])
#test <- rbind(iris3[26:50,,1], iris3[26:50,,2], iris3[26:50,,3])
#cl <- factor(c(rep("s",25), rep("c",25), rep("v",25)))
#knn(train, test, cl, k = 3, prob=TRUE)
attributes(.Last.value)
#train
#test
#cl

#index <- sample(1:nrow(data),round(0.5*nrow(data)))
#train <- data[index,]
#test <- data[-index,]
#train_labels <- train[,2]
#test_labels <- test[,2]
#class_attr <- names(data[2])
#col_names <- names(data[,-2])
#knn_model <- knn(train[,-2],test[,-2],train_labels,k=9,prob=T,use.all = FALSE)
#


























train
