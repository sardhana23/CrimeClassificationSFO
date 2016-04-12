<<<<<<< .mine
#read data
library(randomForest)
library(rpart)
library(mlbench)
library(dplyr)
library(ipred)
library(maboost)
data <- read.csv("C:/Users/Biligiri Vasan/Desktop/Spring 2016/ML/train.csv",colClasses=c("factor","factor","factor","factor","factor","factor"),header = T,sep = ",")
||||||| .r4
data <- read.csv("C:/Users/Biligiri Vasan/Desktop/Spring 2016/ML/train.csv",header = T,sep = ",")
=======
setpwd(.)
data <- read.csv("Dataset/train.csv",header = T,sep = ",")
>>>>>>> .r6
df <- data.frame(data)
<<<<<<< .mine
datatrain <- data[,c(-1,-3,-4,-5,-6,-7)]
classtrain <- data[2]
a <- as.vector(datatrain)
b <- as.vector(classtrain)
str(datatrain)
str(a)
#length(classtrain)
#classtrain[,1]
#head(datatrain[,])
#initial analysis
#data_table <- data.frame(table(df$Category))
#data_ordered <- data_table[order(data_table$Freq,decreasing = T),]
#data_ordered
||||||| .r4
data_table <- data.frame(table(df$Category))
data_ordered <- data_table[order(data_table$Freq,decreasing = T),]
data_ordered
=======
data_table <- data.frame(table(df$Category))
>>>>>>> .r6

<<<<<<< .mine
#data_frame <- select(df,df$Dates,df$DayOfWeek,df$PdDistrict,df$Address,df$X,df$Y)
#data_frame <- df %>% select(df,df$Dates,df$DayOfWeek,df$PdDistrict,df$Address,df$X,df$Y)
||||||| .r4
data_lon <- df$X
=======
############################
#data_ordered <- data_table[order(data_table$Freq,decreasing = T),]
#data_ordered
#
#data_lon <- df$X
#
#print(df$X,digits = 20)
#
#data_add <- df$Address
#
#data_add
############################
>>>>>>> .r6

<<<<<<< .mine
#data.rf <- randomForest(as.formula(names(datatrain[1]),"~.",datatrain,classtrain))
||||||| .r4
print(df$X,digits = 20)
=======
#Formatting Address column - Removing everything upto "Block of"
>>>>>>> .r6

<<<<<<< .mine
#data.bagging <- bagging(df[,c(-2,-3,-6)], df$Category, mfinal=15,
 #                       control=rpart.control(maxdepth=5, minsplit=15))

data.boosting <- maboost(as.formula(paste(names(datatrain[1]),"~.")),datatrain,iter = 10)
#df$Dates + df$DayOfWeek + df$PdDistrict + df$Address + df$X + df$Y

||||||| .r4
data_add <- df$Address

data_add=======
data$Address <- mapply(gsub, "^.*?Block of ","", data["Address"])
>>>>>>> .r6
