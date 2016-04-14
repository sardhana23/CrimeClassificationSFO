library('chron')
data <- read.csv("Dataset/train.csv",header = T,sep = ",")

data$Dates <- as.character(data$Dates)
dtparts = t(as.data.frame(strsplit(data$Dates,' ')))
thetimes = chron(dates=dtparts[,1],times=dtparts[,2],format=c('y-m-d','h:m:s'))

data$Year <- format(thetimes, format="%y")
data$Month <- format(thetimes, format="%m")
data$Date <- format(thetimes, format="%d")
data$hour <- format(thetimes, format="%H")