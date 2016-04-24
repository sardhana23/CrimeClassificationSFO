train

#read data
install.packages('randomForest')
install.packages('rpart')
install.packages('mlbench')
install.packages('dplyr')
install.packages('ipred')
install.packages('maboost')
install.packages('class')
install.packages('chron')
install.packages('adabag')
install.packages('e1071')

library('randomForest')
library('rpart')
library('mlbench')
library('dplyr')
library('ipred')
library('maboost')
library('class')
library('chron')
library('adabag')
library('e1071')

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

#clustering only on X and Y
data <- data_new[c("DayOfWeek","PdDistrict","X","Y","Year","Month","Date","hour","Category")]
data[,1:2]<- sapply(data[,1:2],as.numeric)
data$Category <- factor(data$Category)
cluster_data <- data[3:4]
cl <- kmeans(cluster_data, centers=10000)
#data$PdDistrict <- NULL
data$X <- NULL
data$Y <- NULL

data <- cbind(data,clusternum=cl$cluster)
reorder_data <- data[,c(1,2,3,4,5,6,8,7)]

#from now on use only reorder data to do modeling and analysis
#boosting
sample_data <- createDataPartition(reorder_data$Category, p = .8,list = FALSE,times = 1)
training_data_set<-reorder_data[sample_data,]
test_data_set<-reorder_data[-sample_data,]
train_Class_data<-training_data_set[,7]
test_Class_data<-test_data_set[,7]
class_attr <- names(reorder_data[7])
col_names <- names(reorder_data[,-7])

RF_model <- randomForest(as.formula(paste(names(training_data_set[7]),sep="","~.")),data=training_data_set, importance=TRUE)

boosting_model <- maboost(as.formula(paste(names(training_data_set[7]),sep="","~.")),data=training_data_set,iter =5 , type="discrete")

naive_model <- naiveBayes(as.formula(paste(names(training_data_set[7]),sep="","~.")),data=training_data_set)
naive_result <- predict(naive_model, test_data_set)
