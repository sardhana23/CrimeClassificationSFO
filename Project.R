setpwd(.)
data <- read.csv("Dataset/train.csv",header = T,sep = ",")
df <- data.frame(data)
data_table <- data.frame(table(df$Category))

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

#Formatting Address column - Removing everything upto "Block of"

data$Address <- mapply(gsub, "^.*?Block of ","", data["Address"])
