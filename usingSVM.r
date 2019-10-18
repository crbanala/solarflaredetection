library(e1071)
library(RWeka)

train=read.csv("Train.csv")
test=read.csv("Test.csv")
str(train)


test$K=0
test$L=0
test$M=0
data=rbind(train,test)
str(data)

summary(data)
data$A1=as.factor(data$A1)
data$A2=as.factor(data$A2)
data$A3=as.factor(data$A3)
data$A4=as.factor(data$A4)
data$A5=as.factor(data$A5)
data$A6=as.factor(data$A6)
data$A7=as.factor(data$A7)
data$B1=as.factor(data$B1)
data$B2=as.factor(data$B2)
data$B3=as.factor(data$B3)
data$B4=as.factor(data$B4)
data$B5=as.factor(data$B5)
data$B6=as.factor(data$B6)
data$C1=as.factor(data$C1)
data$C2=as.factor(data$C2)
data$C3=as.factor(data$C3)
data$C4=as.factor(data$C4)
data$D=as.factor(data$D)
data$E=as.factor(data$E)
data$F=as.factor(data$F)
data$G=as.factor(data$G)
data$H=as.factor(data$H)
data$I=as.factor(data$I)
data$J=as.factor(data$J)
data$K=as.factor(data$K)
data$L=as.factor(data$L)
data$M=as.factor(data$M)




test$k = 0
test$l = 0
test$m = 0


fit1 <- svm(K ~ A1+A2+A3+A4+A5+A6+A7+B1+B2+B3+B4+B5+B6+C1+C2+C3+C4+D+E+F+G+H+I+J,data=train)
 pred1=predict(fit1,test)
 test$k=pred1
 
 fit2 <- svm(L ~ season+yr+mnth+holiday+weekday+workingday+weathersit+temp+atemp+hum+windspeed,data=train)
 pred2=predict(fit2,test)
 test$l=pred2

  fit3 <- svm(M ~ season+yr+mnth+holiday+weekday+workingday+weathersit+temp+atemp+hum+windspeed,data=train)
 pred3=predict(fit3,test)
 test$m=pred3

summary(test)
 
test$K=test$k
test$L=test$l
test$M=test$m

i=0:(length(test$m)-1)
final<-data.frame(K=test$K,L=test$L,M=test$M)
write.csv(final,file="output.csv",row.names=FALSE)
