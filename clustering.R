#clustering and attach back the cluster number to data frame
#take 'data' data frame already created from the previous script
data[,1:2]<- sapply(data[,1:2],as.numeric)
data <- data_new[c("DayOfWeek","PdDistrict","X","Y","Year","Month","Date","hour","Category")]
data$Category <- factor(data$Category)
cluster_data <- data[2:4]
cl <- kmeans(cluster_data, centers=10000)
data$PdDistrict <- NULL
data$X <- NULL
data$Y <- NULL

data <- cbind(data,clusternum=cl$cluster)
reorder_data <- data[,c(1,2,3,4,5,7,6)]

#from now on use only reorder data to do modeling and analysis
#boosting
sample_data <- createDataPartition(reorder_data$Category, p = .8,list = FALSE,times = 1)
training_data_set<-reorder_data[sample_data,]
test_data_set<-reorder_data[-sample_data,]
train_Class_data<-training_data_set[,7]
test_Class_data<-test_data_set[,7]
class_attr <- names(reorder_data[7])
col_names <- names(reorder_data[,-7])
boosting_model <- maboost(as.formula(paste(names(training_data_set[7]),sep="","~.")),data=training_data_set,iter =5 , type="discrete")