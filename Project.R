train
=======
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

library('randomForest')
library('rpart')
library('mlbench')
library('dplyr')
library('ipred')
library('maboost')
library('class')
library('chron')
library('adabag')

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
training_data_set<-data_formatted[sample_data,]
test_data_set<-data_formatted[-sample_data,]

form <- as.formula("Category ~ DayOfWeek+PdDistrict+X+Y+Year+Month+Date+hour")
bag_model <- bagging(form, training_data_set, mfinal = 100)


form <- as.formula("Category ~ DayOfWeek+PdDistrict+X+Y+Year+Month+Date+hour")
RF_model <- randomForest(form,data=training_data_set, importance=TRUE)

