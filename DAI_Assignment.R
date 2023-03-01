---
title: "Individual Assignment: Prediction of Shipping Times"
course: "Data Analytics I: Predictive Econometrics"
author: "Arbian Halilaj"
date: "21.01.2022"
---
# R Version last test: 4.0.3
# Load Packages 
library(dplyr)
library(glmnet)
library(GGally)
library(ggplot2)
library(grf)
library(stargazer)
library(funModeling)
library(graphics)
library(reshape2)
library(car)
library(data.table)

# Set working directory
setwd("/Users/arbiun/Desktop/MECONI/2. Semester/DAI/Individual Paper")

# Import datasets
load("olist.Rdata")
load("olist_predict.Rdata")

# Set seed
set.seed(20212022)

############################# Descriptive Statistics  ##########################
# Summary stats
# olist
stargazer(olist, omit.summary.stat = c("p25", "p75"))
df_status1 <- df_status(olist)
stargazer(df_status1[,-(6:7)], summary=FALSE, rownames=FALSE)
# olist_predict
stargazer(as.data.frame(olist_predict), omit.summary.stat = c("p25", "p75"))
df_status2 <- df_status(olist_predict)
stargazer(df_status2[,-(6:7)], summary=FALSE, rownames=FALSE)

# Histograms of numerical variables
hist(olist$shippingtime)
hist(olist$payment_value)
hist(olist$weight_g)
hist(olist$freight_rate)
hist(olist$distance)

############################# Data Cleansing ###################################

# Adressing issues
# Set negative values in shippingtimes as NA
olist$shippingtime <- replace(olist$shippingtime, which(olist$shippingtime <= 0), NA)

# Set zero values in weight_g, freight_rate & distance as NA
olist$weight_g <- replace(olist$weight_g, which(olist$weight_g == 0), NA)
olist_predict$weight_g <- replace(olist_predict$weight_g, which(olist_predict$weight_g == 0), NA)

olist$freight_rate <- replace(olist$freight_rate, which(olist$freight_rate == 0), NA)
olist_predict$freight_rate <- replace(olist_predict$freight_rate, which(olist_predict$freight_rate == 0), NA)

olist$distance <- replace(olist$distance, which(olist$distance == 0), NA)
olist_predict$distance <- replace(olist_predict$distance, which(olist_predict$distance == 0), NA)

# Remove NA's 
olist <- na.omit(olist) #(77660-75494)
olist_predict <- na.omit(olist_predict)

# Logarithmize numerical variables which are relatively large and store in separate df
# Normally, only the dependent variable is transformed:
olist_log <- olist
olist_log$shippingtime <- log(olist_log$shippingtime)

# Rooted dependent variable
olist_root <- olist
olist_root$shippingtime <- sqrt(olist_root$shippingtime)

# Since also the distributions of the numerical independent variables is highly skewed
# we logarithmize in a separate df
olist_log_all <- olist
olist_log_all$shippingtime <- log(olist_log_all$shippingtime)
hist(olist_log_all$shippingtime)
olist_log_all$payment_value <- log(olist_log_all$payment_value)
hist(olist_log_all$payment_value)
olist_log_all$weight_g <- log(olist_log_all$weight_g)
hist(olist_log_all$weight_g)
olist_log_all$freight_rate <- log(olist_log_all$freight_rate)
hist(olist_log_all$freight_rate)
olist_log_all$distance <- log(olist_log_all$distance)
hist(olist_log_all$distance)

############################# Correlation Analysis #############################

# Correlation Analysis
sds <- olist
sds[] <- lapply(sds,as.numeric)
cormatweek <- round(cor(sds, method = "spearman"),2)

# get upper triangle of the correlation matrix
get_upper_tri_week <- function(cormatweek){
  cormatweek[lower.tri(cormatweek)] <- NA
  return(cormatweek)
}

upper_tri_week <- get_upper_tri_week(cormatweek)

# upper_tri_week
melted_cormat_week <- melt(upper_tri_week, na.rm = TRUE)

ggheatmap <- ggplot(data = melted_cormat_week, aes(Var2, Var1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 8, hjust = 1)) +
  coord_fixed() +
  theme(axis.text.y = element_text(vjust = 1, 
                                   size = 8, hjust = 1))
# add numbers
ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 3) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.75),
    legend.direction = "horizontal") +
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

###############################################################################
# Partition of the data
sample75 <- sample.int(n = nrow(olist), size = floor(.75*nrow(olist)), replace = F)
#sample80 <- sample.int(n = nrow(olist), size = floor(.8*nrow(olist)), replace = F)

# First Data Set (No Log Transformation)
train <- olist[sample75, ]
test  <- olist[-sample75, ]
#train <- olist[sample80, ]
#test  <- olist[-sample80, ]

x0 <- train[,-1]
x0 <- x0[,-15]
x0 <- as.matrix(x0)
dummies <- train[,16] # transform character variable into dummies
dummies <- fastDummies::dummy_cols(dummies)
dummies <- dummies[,-1]
dummies <- dummies[,-62]
a <- names(dummies)
dummies <- as.matrix(dummies)
x1 <- cbind(x0, dummies)
x2 <- x0[,-5] # remove uncorrelated variables
x2 <- x2[,-(8:9)] # remove uncorrelated variables
y <- train[,1] 
y <- as.matrix(y)

x0_test <- test[,-1]
x0_test <- x0_test[,-15]
x0_test <- as.matrix(x0_test)
dummies2 <- test[,16] # transform character variable into dummies
dummies2 <- fastDummies::dummy_cols(dummies2)
dummies2 <- dummies2[,-1]
b <- names(dummies2)
diff <- setdiff(a, b) #remove product_category_major_security_and_services in dummies (above)
x1_test <- cbind(x0_test, dummies2)
y_test <- as.matrix(test[,1])

# Second Data Set (Y Log Transformation)

train_log <- olist_log[sample75, ]
test_log  <- olist_log[-sample75, ]
#train_log <- olist_log[sample80, ]
#test_log  <- olist_log[-sample80, ]

y_log <- train_log[,1]
y_log <- as.matrix(y_log)


# Second Data Set (Y & X Log Transformation)

train_log_all <- olist_log_all[sample75, ]
test_log_all  <- olist_log_all[-sample75, ]
#train_log_all <- olist_log_all[sample80, ]
#test_log_all  <- olist_log_all[-sample80, ]

x_log <- train_log_all[,-1]
x_log <- x_log[,-15]
x_log <- as.matrix(x_log)

x_test_log <- test_log_all[,-1]
x_test_log <- x_test_log[,-15]

################################ OLS Regression ################################
df0 <- data.frame(cbind(y,x0))
ols0 <- lm(df0$shippingtime~., data = df0)

df1 <- data.frame(cbind(y,x1)) 
ols1 <- lm(df1$shippingtime~., data = df1)

df2 <- data.frame(cbind(y_log,x0)) 
ols2 <- lm(df2$shippingtime~., data = df2)

df3 <- data.frame(cbind(y_log,x_log)) 
ols3 <- lm(df3$shippingtime~., data = df3)

df4 <- data.frame(cbind(y,x2)) 
ols4 <- lm(df4$shippingtime~., data = df4)

summary(ols0)
summary(ols1)
summary(ols2)
summary(ols3)
summary(ols4)

# Diagnostics
# par(mfrow=c(2,2))
plot(ols0, 2)
plot(ols1, 2)
plot(ols2, 2)
plot(ols3, 2)
plot(ols4, 2)

vif(ols0)
vif(ols1)
vif(ols2)
vif(ols3)
vif(ols4)

# Predict test data with OLS
pred_ols0 <- predict(ols0, newdata = data.frame(x0_test))
pred_ols1 <- predict(ols1, newdata = data.frame(x1_test))
pred_ols2 <- predict(ols2, newdata = data.frame(x0_test))
pred_ols2 <- exp(pred_ols2)
pred_ols3 <- predict(ols3, newdata = data.frame(x_test_log))
pred_ols3 <- exp(pred_ols3)
pred_ols4 <- predict(ols4, newdata = data.frame(x0_test))

# R-squared out of sample
SST <- sum((y_test - mean(y_test))^2) 
SSE <- sum((y_test - pred_ols0)^2)
r2_ols0 <- 1 - SSE/SST
print(paste("The R-squared for the ols predictions is: ", round(r2_ols0, 4)))
SSE <- sum((y_test - pred_ols1)^2)
r2_ols1 <- 1 - SSE/SST
print(paste("The R-squared for the ols predictions is: ", round(r2_ols1, 4)))
SSE <- sum((y_test - pred_ols2)^2)
r2_ols2 <- 1 - SSE/SST
print(paste("The R-squared for the ols predictions is: ", round(r2_ols2, 4)))
SSE <- sum((y_test - pred_ols3)^2)
r2_ols3 <- 1 - SSE/SST
print(paste("The R-squared for the ols predictions is: ", round(r2_ols3, 4)))
SSE <- sum((y_test - pred_ols4)^2)
r2_ols4 <- 1 - SSE/SST
print(paste("The R-squared for the ols predictions is: ", round(r2_ols4, 4)))

################################ Loss Regressions ##############################

# Optimal Alpha (0 = ridge, 0.5 = elastic net, 1 = lasso)
# Without Character Data
# Loop 
for (i in 0:10) {
  assign(paste("fit", i, sep=""), cv.glmnet(x0, y, type.measure="mse", nfolds = 5,
                                            alpha=i/10, family="gaussian"))
}

yhat0 <- predict(fit0, s=fit0$lambda.min, newx=x0_test)
yhat1 <- predict(fit1, s=fit1$lambda.min, newx=x0_test)
yhat2 <- predict(fit2, s=fit2$lambda.min, newx=x0_test)
yhat3 <- predict(fit3, s=fit3$lambda.min, newx=x0_test)
yhat4 <- predict(fit4, s=fit4$lambda.min, newx=x0_test)
yhat5 <- predict(fit5, s=fit5$lambda.min, newx=x0_test)
yhat6 <- predict(fit6, s=fit6$lambda.min, newx=x0_test)
yhat7 <- predict(fit7, s=fit7$lambda.min, newx=x0_test)
yhat8 <- predict(fit8, s=fit8$lambda.min, newx=x0_test)
yhat9 <- predict(fit9, s=fit9$lambda.min, newx=x0_test)
yhat10 <- predict(fit10, s=fit10$lambda.min, newx=x0_test)

mse0 <- mean((y_test - yhat0)^2)
mse1 <- mean((y_test - yhat1)^2)
mse2 <- mean((y_test - yhat2)^2)
mse3 <- mean((y_test - yhat3)^2)
mse4 <- mean((y_test - yhat4)^2)
mse5 <- mean((y_test - yhat5)^2)
mse6 <- mean((y_test - yhat6)^2)
mse7 <- mean((y_test - yhat7)^2)
mse8 <- mean((y_test - yhat8)^2)
mse9 <- mean((y_test - yhat9)^2)
mse10 <- mean((y_test - yhat10)^2)

# optimal alpha = 0.7
plot(fit4, main="Loss (0.7)")
coef(fit5, s = "lambda.min")

# R-squared out of sample
SSE <- sum((y_test - yhat4)^2)
r2_loss0 <- 1 - SSE/SST
print(paste("The R-squared for the loss predictions is: ", round(r2_loss0, 4)))

# With Character Data
# Loop 
for (i in 0:10) {
  assign(paste("fit", i, sep=""), cv.glmnet(x1, y, type.measure="mse", nfolds = 5,
                                            alpha=i/10, family="gaussian"))
}

x1_test <- data_frame(x1_test)
yhat0 <- predict(fit0, s=fit0$lambda.min, newx=x1_test)
yhat1 <- predict(fit1, s=fit1$lambda.min, newx=x1_test)
yhat2 <- predict(fit2, s=fit2$lambda.min, newx=x1_test)
yhat3 <- predict(fit3, s=fit3$lambda.min, newx=x1_test)
yhat4 <- predict(fit4, s=fit4$lambda.min, newx=x1_test)
yhat5 <- predict(fit5, s=fit5$lambda.min, newx=x1_test)
yhat6 <- predict(fit6, s=fit6$lambda.min, newx=x1_test)
yhat7 <- predict(fit7, s=fit7$lambda.min, newx=x1_test)
yhat8 <- predict(fit8, s=fit8$lambda.min, newx=x1_test)
yhat9 <- predict(fit9, s=fit9$lambda.min, newx=x1_test)
yhat10 <- predict(fit10, s=fit10$lambda.min, newx=x1_test)

mse0 <- mean((y_test - yhat0)^2)
mse1 <- mean((y_test - yhat1)^2)
mse2 <- mean((y_test - yhat2)^2)
mse3 <- mean((y_test - yhat3)^2)
mse4 <- mean((y_test - yhat4)^2)
mse5 <- mean((y_test - yhat5)^2)
mse6 <- mean((y_test - yhat6)^2)
mse7 <- mean((y_test - yhat7)^2)
mse8 <- mean((y_test - yhat8)^2)
mse9 <- mean((y_test - yhat9)^2)
mse10 <- mean((y_test - yhat10)^2)

# optimal alpha = 0.6
plot(fit5, main="Elastic Net")
coef(fit5, s = "lambda.min")

# R-squared out of sample
SSE <- sum((y_test - yhat5)^2)
r2_loss1 <- 1 - SSE/SST
print(paste("The R-squared for the loss predictions is: ", round(r2_loss1, 4)))


####################### Category Variable Reduction ###########################
# I tried to explore the categories in order to find a way to reduce the dimensionality
# Unfortunately, this analysis was not fruitful when applied to ls regression
# Any reprogramming of the variable done below did not improve ls regression
# The first two approaches try to create a new dummy which is equal to 1 if a category produces outliers, 0 else
# The last approach just uses a dummy of each category which stood out in terms of outliers

table(train$product_category_major)

reorder_size <- function(x) {
        factor(x, levels = names(sort(table(x), decreasing = TRUE)))
}
         
ggplot(train, aes(x = reorder_size(product_category_major))) +
  geom_bar(aes(y = (..count..)/sum(..count..))) +
  scale_y_continuous(labels = scales::percent, name = "Proportion") +
  xlab("Product Category") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(train, aes(x = product_category_major, y = shippingtime)) +
  geom_boxplot() + 
  xlab("Product category") + ylab("Shipping time") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# First approach:
# At least two outliers above 100 days
# Autos, bed_bath_table, computers_accessories, garden_tools, health_beauty, housewares, perfumery, sports_leisure, telephony, toys

df <- train%>%
  mutate(product_category_major=case_when(
    product_category_major=="auto" ~ 1,
    product_category_major=="bed_bath_table" ~ 1,
    product_category_major=="computers_accessories" ~ 1,
    product_category_major=="garden_tools" ~ 1,
    product_category_major=="health_beauty" ~ 1,
    product_category_major=="housewares" ~ 1,
    product_category_major=="perfumery" ~ 1,
    product_category_major=="sports_leisure" ~ 1,
    product_category_major=="telephony" ~ 1,
    product_category_major=="toys" ~ 1,
  ))

df <- df %>%
  mutate(product_category_major = if_else(is.na(product_category_major), 0, product_category_major))

hist(df$product_category_major)

fit1 <- lm(df$shippingtime~df$product_category_major, data = df)                                   
summary(fit1)

fit2 <- lm(df$shippingtime~., data = df)                                   
summary(fit2)

# Second approach: another try with more categories
df1 <- train%>%
  mutate(product_category_major=case_when(
    product_category_major=="auto" ~ 1,
    product_category_major=="baby" ~ 1,
    product_category_major=="bed_bath_table" ~ 1,
    product_category_major=="computers_accessories" ~ 1,
    product_category_major=="cool_stuff" ~ 1,
    product_category_major=="electronics" ~ 1,
    product_category_major=="furniture_decor" ~ 1,
    product_category_major=="garden_tools" ~ 1,
    product_category_major=="health_beauty" ~ 1,
    product_category_major=="housewares" ~ 1,
    product_category_major=="office_furniture" ~ 1,
    product_category_major=="perfumery" ~ 1,
    product_category_major=="sports_leisure" ~ 1,
    product_category_major=="stationery" ~ 1,
    product_category_major=="telephony" ~ 1,
    product_category_major=="toys" ~ 1,
    product_category_major=="watches_gifts" ~ 1
  ))

df1 <- df1 %>%
  mutate(product_category_major = if_else(is.na(product_category_major), 0, product_category_major))

hist(df1$product_category_major)

fit3 <- lm(df$shippingtime~df$product_category_major, data = df1)                                   
summary(fit3)

fit4 <- lm(df$shippingtime~., data = df1)                                   
summary(fit4)

# Third approach: seperate dummies for a collection of categories

df2 <- train[,-16]
df2$auto <- ifelse(train$product_category_major == "auto", 1, 0)
df2$baby <- ifelse(train$product_category_major == "baby", 1, 0)
df2$bedbathtable <- ifelse(train$product_category_major == "bed_bath_table", 1, 0)
df2$computeraccessories <- ifelse(train$product_category_major == "computers_accessories", 1, 0)
df2$coolstuff <- ifelse(train$product_category_major == "cool_stuff", 1, 0)
df2$electronics <- ifelse(train$product_category_major == "electronics", 1, 0)
df2$furnituredecor <- ifelse(train$product_category_major == "furniture_decor", 1, 0)
df2$gardentools <- ifelse(train$product_category_major == "garden_tools", 1, 0)
df2$healthbeauty <- ifelse(train$product_category_major == "health_beauty", 1, 0)
df2$houseware <- ifelse(train$product_category_major == "houseware", 1, 0)
df2$officefurniture <- ifelse(train$product_category_major == "office_furniture", 1, 0)
df2$perfumery <- ifelse(train$product_category_major == "perfumery", 1, 0)
df2$sportsleisure <- ifelse(train$product_category_major == "sports_leisure", 1, 0)
df2$stationery <- ifelse(train$product_category_major == "stationery", 1, 0)
df2$telephony <- ifelse(train$product_category_major == "telephony", 1, 0)
df2$toys <- ifelse(train$product_category_major == "toys", 1, 0)
df2$watchesgifts <- ifelse(train$product_category_major == "watches_gifts", 1, 0)

df3 <- df2[,-(2:15)]

fit5 <- lm(df3$shippingtime~., data = df3)                                   
summary(fit5)

fit6 <- lm(df2$shippingtime~., data = df2)                                 
summary(fit6)

################## Category Variable Reduction: Second attempt #################
# This approach experiments with the idea of post-loss regression
# Thereby, we want to reduce the number of categories
# In a second approach we try out gglasso for grouped lasso regression

xx <- train[,16]
xx <- fastDummies::dummy_cols(xx)
xx <- xx[,-1]
xx <- xx[,-62]
xx <- as.matrix(xx)
yy <- train[,1]
yy <-  as.matrix(yy)

xx_test <- test[,16]
xx_test <- fastDummies::dummy_cols(xx_test)
xx_test <- xx_test[,-1]
xx_test = as.matrix(xx_test)
product_category_major_security_and_services <- matrix(0, 18874, 1)
xx_test <- cbind(xx_test, product_category_major_security_and_services)

for (i in 0:10) {
  assign(paste("fitd", i, sep=""), cv.glmnet(xx, yy, type.measure="mse", 
                                            alpha=i/10,family="gaussian"))
}

yhat0 <- predict(fitd0, s=fitd1$lambda.1se, newx=xx_test)
yhat1 <- predict(fitd1, s=fitd1$lambda.1se, newx=xx_test)
yhat2 <- predict(fitd2, s=fitd2$lambda.1se, newx=xx_test)
yhat3 <- predict(fitd3, s=fitd3$lambda.1se, newx=xx_test)
yhat4 <- predict(fitd4, s=fitd4$lambda.1se, newx=xx_test)
yhat5 <- predict(fitd5, s=fitd5$lambda.1se, newx=xx_test)
yhat6 <- predict(fitd6, s=fitd6$lambda.1se, newx=xx_test)
yhat7 <- predict(fitd7, s=fitd7$lambda.1se, newx=xx_test)
yhat8 <- predict(fitd8, s=fitd8$lambda.1se, newx=xx_test)
yhat9 <- predict(fitd9, s=fitd9$lambda.1se, newx=xx_test)
yhat10 <- predict(fitd10, s=fitd10$lambda.1se, newx=xx_test)

mse0 <- mean((y_test - yhat0)^2)
mse1 <- mean((y_test - yhat1)^2)
mse2 <- mean((y_test - yhat2)^2)
mse3 <- mean((y_test - yhat3)^2)
mse4 <- mean((y_test - yhat4)^2)
mse5 <- mean((y_test - yhat5)^2)
mse6 <- mean((y_test - yhat6)^2)
mse7 <- mean((y_test - yhat7)^2)
mse8 <- mean((y_test - yhat8)^2)
mse9 <- mean((y_test - yhat9)^2)
mse10 <- mean((y_test - yhat10)^2)

# Optimal alpha = 0.1
fit.ridge2 <- glmnet(xx, yy, family="gaussian", alpha=.1)

plot(fit.ridge2, xvar="lambda", label=TRUE)

ridge.cv2 <- cv.glmnet(xx, yy, type.measure = "mse", nfolds = 5, family = "gaussian", alpha = 0.1)

plot(ridge.cv2, main="Ridge (0.1)")

pred_ridge2 <- predict(ridge.cv2, newx = xx_test, s = ridge.cv2$lambda.min)

coef_ridge2 <- coef(ridge.cv2, s = "lambda.min")
names <- coef_ridge2@i
coef_ridge2 <- coef_ridge2@x
coef_ridge2 <- data.frame(coef_ridge2)
coef_ridge2 <- coef_ridge2[-1,]
coef_ridge2 <- data.frame(coef_ridge2)
rownames(coef_ridge2) <- names[-1]
coef_ridge2 <- round(coef_ridge2, 6)

coef20 <- coef_ridge2 %>%
  arrange(desc(abs(coef_ridge2))) %>%
  slice(1:20)

z <- c(29, 3, 9, 23, 46, 58, 18, 2, 52, 32, 63, 47, 24, 50, 61, 36, 12, 49, 37, 42)

coef20_df <- xx[,z]

# Performance measures
mse_ridge2 <- mean((y_test - pred_ridge2)^2)

# Post-Loss Regression
columnname <- rownames(coef(ridge.cv2, s="lambda.min"))[coef(ridge.cv2, s="lambda.min")[,1]!=0]
columnname <- columnname[-1]
data <- subset(xx, select = columnname)
data <- data.frame(cbind(y,x0,data))
df_test <- subset(x1_test, select = columnname)
df_test <- cbind(df_test, x0_test)
df_test <- data.frame(df_test)

ols_postloss <- lm(data$shippingtime~., data = data)
summary(ols_postloss)

plot(ols_postloss, 2)
vif(ols_postloss)

pred_ols_postloss <- predict(ols_postloss, newdata = df_test)

SSE <- sum((y_test - pred_ols_postloss)^2)
r2_ols_postloss <- 1 - SSE/SST
print(paste("The R-squared for the post loss ols predictions is: ", round(r2_ols_postloss, 4)))

# Post-Loss y log Regression
data <- data[,-1]
data <- data.frame(cbind(y_log, data))

ols_postloss_log <- lm(data$shippingtime~., data = data)
summary(ols_postloss_log)

plot(ols_postloss_log, 2)
vif(ols_postloss_log)

pred_ols_postloss_log <- predict(ols_postloss_log, newdata = df_test)

pred_ols_postloss_log <- exp(pred_ols_postloss_log)

SSE <- sum((y_test - pred_ols_postloss_log)^2)
r2_ols_postloss_log <- 1 - SSE/SST
print(paste("The R-squared for the post loss ols predictions is: ", round(r2_ols_postloss_log, 4)))

# Grouped Lasso Regression
# load gglasso library
library(gglasso)

# define group index
group1 <- rep(1:9.25,each=8)

# fit group lasso penalized least squares
gl <- gglasso(x=xx,y=yy,group=group1,loss="ls")
plot(gl)

# 5-fold cross validation using group lasso
# penalized least square regression
gl_cv <- cv.gglasso(x=xx, y=yy, group=group1, loss="ls", nfolds=5)
plot(gl_cv)
coef(gl_cv, s = "lambda.min")

################################ Random Forest #############################

## Initial step: analyzing optimal forest size

# Data
x <- olist
dummy <- x[,16]
dummy <- fastDummies::dummy_cols(dummy)
dummy <- dummy[,-1]
x <- x[,-1]
x <- x[,-15]
x <- cbind(x, dummy)
x <- as.matrix(x)
y <- olist[,1]
y <- as.matrix(y)

#x <- df2[,-1] #choosen dummies 

# Generate variable with the rows in training data
size <- ceiling(0.5 * nrow(olist))
training_set <- sample(seq_len(nrow(olist)), size = size, replace = FALSE)

rep <- 1000 # number of trees
cov <- floor(sqrt(ncol(olist))) # share of covariates
frac <- 1/2 # fraction of subsample
min_obs <- 500 # minimum size of terminal leaves in trees

# R2 - trees graph
sizes <- c(1500, 1250, 1000, 750, 500, 400, 300, 200, 100, 75, 50, 40, 30, 20, 10, 5, 4, 3, 2, 1) # Select a grid of forest sizes

# Prepare matrix to store results
tree_graph <- matrix(NA, nrow = length(sizes), ncol = 3)
colnames(tree_graph) <- c("Trees", "R2", "Marginal R2")

# assign sizes

tree_graph[, 1] <- sizes
# Sum of Squares Total (not variable within the loop)
SST <- sum(((y[-training_set, ]) - (mean(y[-training_set, ])))^2)

# start loop

for (tree_idx in sizes){
  # Estimate Forests
  forest <- regression_forest(x[training_set, ], (y[training_set, ]),
                              mtry = cov, sample.fraction = frac, num.trees = tree_idx, 
                              min.node.size = min_obs, honesty = FALSE)
  # prediction in test sample
  fit <- predict(forest, newdata = x[-training_set, ])$predictions
  
  # store R-squared
  tree_graph[tree_graph[, 1] == tree_idx, 2] <- 1 - sum(((y[-training_set, ]) - fit)^2)/SST
}

# fill in values for marginal R2
tree_graph[, 3] <- tree_graph[, 2] - rbind(as.matrix(tree_graph[-1, 2]), tree_graph[nrow(tree_graph), 2])

# R2
plot(tree_graph[, 1], tree_graph[, 2], type = "o", xlab = "Trees", ylab = "R-squared", main = "Out-of-Sample Accuracy")
abline(a = tree_graph[1, 2], b = 0, col = "red")

# Marginal R2
plot(tree_graph[, 1], tree_graph[, 3], type = "o", xlab = "Trees", ylab = "Delta R-squared", main = "Marginal Out- of-Sample Accuracy")
abline(a = 0, b = 0, col = "red")

## Random Forest ###############################################################

# Build Forest
rep <- 1000 # number of trees
cov <- floor(sqrt(ncol(olist_log))) # share of covariates
frac <- 1/2 # fraction of subsample
min_obs <- 500 # minimum size of terminal leaves in trees

forest_basic <- regression_forest(x[training_set, ], y[training_set, ],
                                mtry = cov, sample.fraction = frac, num.trees = rep, 
                                min.node.size = min_obs, honesty = FALSE)

# Plot a tree of the forest
#tree <- get_tree(forest_basic, 1) 
#plot(tree)

# Prediction
pred_tree <- predict(forest_basic, newdata = x[-training_set, ])$predictions

# R-squared out of sample
SST <- sum(((y[-training_set, ]) - mean((y[-training_set, ])))^2) 
SSE_forest <- sum(((y[-training_set, ]) - pred_tree)^2)
r2_forest <- 1 - SSE_forest/SST
print(paste("The R-squared for the forest predictions is: ", round(r2_forest, 4)))

## Choosen Dummies Forest

df2 <- olist[,-16]
df2$auto <- ifelse(olist$product_category_major == "auto", 1, 0)
df2$baby <- ifelse(olist$product_category_major == "baby", 1, 0)
df2$bedbathtable <- ifelse(olist$product_category_major == "bed_bath_table", 1, 0)
df2$computeraccessories <- ifelse(olist$product_category_major == "computers_accessories", 1, 0)
df2$coolstuff <- ifelse(olist$product_category_major == "cool_stuff", 1, 0)
df2$electronics <- ifelse(olist$product_category_major == "electronics", 1, 0)
df2$furnituredecor <- ifelse(olist$product_category_major == "furniture_decor", 1, 0)
df2$gardentools <- ifelse(olist$product_category_major == "garden_tools", 1, 0)
df2$healthbeauty <- ifelse(olist$product_category_major == "health_beauty", 1, 0)
df2$houseware <- ifelse(olist$product_category_major == "houseware", 1, 0)
df2$officefurniture <- ifelse(olist$product_category_major == "office_furniture", 1, 0)
df2$perfumery <- ifelse(olist$product_category_major == "perfumery", 1, 0)
df2$sportsleisure <- ifelse(olist$product_category_major == "sports_leisure", 1, 0)
df2$stationery <- ifelse(olist$product_category_major == "stationery", 1, 0)
df2$telephony <- ifelse(olist$product_category_major == "telephony", 1, 0)
df2$toys <- ifelse(olist$product_category_major == "toys", 1, 0)
df2$watchesgifts <- ifelse(olist$product_category_major == "watches_gifts", 1, 0)

x <- df2[,-1]
x <- x[,-(9:10)]

forest_basic2 <- regression_forest(x[training_set, ], y[training_set, ],
                                  mtry = cov, sample.fraction = frac, num.trees = rep, 
                                  min.node.size = min_obs, honesty = FALSE)

# Plot a tree of the forest
tree <- get_tree(forest_basic2, 1) 
plot(tree)

# Prediction
pred_tree2 <- predict(forest_basic2, newdata = x[-training_set, ])$predictions

# R-squared out of sample
SST <- sum(((y[-training_set, ]) - mean((y[-training_set, ])))^2) 
SSE_forest2 <- sum(((y[-training_set, ]) - pred_tree2)^2)
r2_forest2 <- 1 - SSE_forest2/SST
print(paste("The R-squared for the forest predictions is: ", round(r2_forest2, 4)))

## Forest Deep

min_obs <- 5 # minimum size of terminal leaves in trees

forest_deep <- regression_forest(x[training_set, ], y[training_set, ],
                                  mtry = cov, sample.fraction = frac, num.trees = rep, 
                                  min.node.size = min_obs, honesty = FALSE)

# Plot a tree of the forest
#tree <- get_tree(forest_deep, 1) 
#plot(tree)

# Prediction
pred_tree_deep <- predict(forest_deep, newdata = x[-training_set, ])$predictions

# R-squared out of sample
SST <- sum(((y[-training_set, ]) - mean((y[-training_set, ])))^2) 
SSE_forest_deep <- sum(((y[-training_set, ]) - pred_tree_deep)^2)
r2_forest_deep <- 1 - SSE_forest_deep/SST
print(paste("The R-squared for the forest predictions is: ", round(r2_forest_deep, 4)))

## Random Forest with log transformed y ########################################

x_log = olist_log[,-1]
x_log = x_log[,-15]
x_log = x_log[,-(9:10)] # removing uncorrelated variables to improve prediction
x_log = x_log[,-5] # removing uncorrelated variables to improve prediction
x_log = as.matrix(x_log)
y_log = olist_log[,1]
y_log = as.matrix(y_log)

# Build Forest Log
set.seed(20212022)
min_obs <- 500 # minimum size of terminal leaves in trees

forest_log <- regression_forest(x_log[training_set, ], y_log[training_set, ],
                                  mtry = cov, sample.fraction = frac, num.trees = rep, 
                                  min.node.size = min_obs, honesty = FALSE)

# Plot a tree of the forest
#tree <- get_tree(forest_log, 1) 
#plot(tree)

# Prediction
pred_tree_log <- predict(forest_log, newdata = x_log[-training_set, ])$predictions

pred_tree_exp <- exp(pred_tree_log)

# R-squared out of sample
SST <- sum(((exp(y_log[-training_set, ])) - mean((exp(y_log[-training_set, ]))))^2) 
SSE_forest_log <- sum(((exp(y_log[-training_set, ])) - pred_tree_exp)^2)
r2_forest_log <- 1 - SSE_forest_log/SST
print(paste("The R-squared for the forest predictions is: ", round(r2_forest_log, 4)))

## Deep Forest with log transformed y

# Build Forest Log
min_obs <- 5 # minimum size of terminal leaves in trees

forest_deep_log <- regression_forest(x_log[training_set, ], y_log[training_set, ],
                                mtry = cov, sample.fraction = frac, num.trees = rep, 
                                min.node.size = min_obs, honesty = FALSE)

# Plot a tree of the forest
#tree <- get_tree(forest_log, 1) 
#plot(tree)

# Prediction
pred_tree_deep_log <- predict(forest_deep_log, newdata = x_log[-training_set, ])$predictions

pred_tree_deep_exp <- exp(pred_tree_deep_log)

# R-squared out of sample
mse <- mean((exp(y_log[-training_set, ]) - pred_tree_deep_exp)^2)
SST <- sum(((exp(y_log[-training_set, ])) - mean((exp(y_log[-training_set, ]))))^2) 
SSE_forest_deep_log <- sum(((exp(y_log[-training_set, ])) - pred_tree_deep_exp)^2)
r2_forest_deep_log <- 1 - SSE_forest_deep_log/SST
print(paste("The R-squared for the forest predictions is: ", round(r2_forest_deep_log, 4)))

## Deep Forest with log transformed y and choosen dummies

# Build Deep Forest Log
min_obs <- 5 # minimum size of terminal leaves in trees

x_log1 <- cbind(x_log, x)

forest_deep_log2 <- regression_forest(x_log1[training_set, ], y_log[training_set, ],
                                     mtry = cov, sample.fraction = frac, num.trees = rep, 
                                     min.node.size = min_obs, honesty = FALSE)

# Plot a tree of the forest
#tree <- get_tree(forest_log, 1) 
#plot(tree)

# Prediction
pred_tree_deep_log2 <- predict(forest_deep_log2, newdata = x_log1[-training_set, ])$predictions

pred_tree_deep_exp2 <- exp(pred_tree_deep_log2)

# R-squared out of sample
SST <- sum(((exp(y_log[-training_set, ])) - mean((exp(y_log[-training_set, ]))))^2) 
SSE_forest_deep_log2 <- sum(((exp(y_log[-training_set, ])) - pred_tree_deep_exp2)^2)
r2_forest_deep_log2 <- 1 - SSE_forest_deep_log2/SST
print(paste("The R-squared for the forest predictions is: ", round(r2_forest_deep_log2, 4)))

## Last approach: ###############################################################

#df5 <- olist[,-(1:16)]
#df5$fashion_bags_accessories <- ifelse(olist$product_category_major == "fashion_bags_accessories", 1, 0)
#df5$art <- ifelse(olist$product_category_major == "art", 1, 0)
#df5$books_general_interest <- ifelse(olist$product_category_major == "books_general_interest", 1, 0)
#df5$construction_tools_tools <- ifelse(olist$product_category_major == "construction_tools_tools", 1, 0)
#df5$home_appliances_2 <- ifelse(olist$product_category_major == "home_appliances_2", 1, 0)
#df5$office_furniture <- ifelse(olist$product_category_major == "office_furniture", 1, 0)
#df5$construction_tools_construction <- ifelse(olist$product_category_major == "construction_tools_construction", 1, 0)
#df5$air_conditioning <- ifelse(olist$product_category_major == "air_conditioning", 1, 0)
#df5$kitchen_dining_laundry_garden_furniture <- ifelse(olist$product_category_major == "kitchen_dining_laundry_garden_furniture", 1, 0)
#df5$fashion_shoes <- ifelse(olist$product_category_major == "fashion_shoes", 1, 0)
#df5$small_appliances <- ifelse(olist$product_category_major == "small_appliances", 1, 0)
#df5$home_comfort_2 <- ifelse(olist$product_category_major == "home_comfort_2", 1, 0)
#df5$diapers_hygiena <- ifelse(olist$product_category_major == "diapers_hygiena", 1, 0)
#df5$housewares <- ifelse(olist$product_category_major == "housewares", 1, 0)
#df5$pet_shop <- ifelse(olist$product_category_major == "pet_shop", 1, 0)
#df5$flowers <- ifelse(olist$product_category_major == "flowers", 1, 0)
#df5$cdc_dvds_musicals <- ifelse(olist$product_category_major == "cdc_dvds_musicals", 1, 0)
#df5$home_construction <- ifelse(olist$product_category_major == "home_construction", 1, 0)
#df5$furniture_mattress_and_upholstery <- ifelse(olist$product_category_major == "furniture_mattress_and_upholstery", 1, 0)
#df5$food <- ifelse(olist$product_category_major == "food", 1, 0)
#x <- df5[,-1]

df5 <- olist[,-(1:16)]
x <- cbind(x_log, df5)
x <- x[,-(9:10)]
x <- x[,-(9)]
x <- x[,-(5)]
x <- x[,-(2)]
x <- x[,-(1)]

y_root <- olist_root[,1]
y_root <- as.matrix(y_root)

# Generate variable with the rows in training data
size <- ceiling(0.75 * nrow(olist))
training_set <- sample(seq_len(nrow(olist)), size = size, replace = FALSE)

# Build Forest Log
rep <- 1000 # number of trees
cov <- floor(sqrt(ncol(olist_log))) # share of covariates
frac <- 1/2 # fraction of subsample
min_obs <- 5 # minimum size of terminal leaves in trees

forest_deep_root <- regression_forest(x[training_set, ], y_root[training_set, ],
                                     mtry = cov, sample.fraction = frac, num.trees = rep, 
                                     min.node.size = min_obs, honesty = FALSE)

# Plot a tree of the forest
#tree <- get_tree(forest_deep_log, 1) 
#plot(tree)

# Prediction
pred_forest_deep_root <- predict(forest_deep_root, newdata = x[-training_set, ])$predictions

pred_forest_deep_root <- (pred_forest_deep_root)^2

y_root <- (y_root)^2

# R-squared out of sample
mse <- mean((y_root[-training_set, ] - pred_forest_deep_root)^2)
SST <- sum(((y_root[-training_set, ]) - mean((y_root[-training_set, ])))^2) 
SSE_forest_deep_root <- sum(((y_root[-training_set, ]) - pred_forest_deep_root)^2)
r2_forest_deep_root <- 1 - SSE_forest_deep_root/SST
print(paste("The R-squared for the forest predictions is: ", round(r2_forest_deep_root, 4)))

################################ Performance of the models #####################
r2_all <- as.data.frame(rbind(r2_ols0, r2_ols1, r2_ols2, r2_ols3, r2_ols4, 
                              r2_ols_postloss, r2_ols_postloss_log,  
                              r2_loss0, r2_loss1, 
                              r2_forest, r2_forest2, r2_forest_deep, r2_forest_log,
                              r2_forest_deep_log, r2_forest_deep_log2, r2_forest_deep_root))
print(r2_all)

################################ Out-of-sample Prediction #######################
olist_predict <- olist_predict[,c(3,4,7,8,11)]

pred_final <- predict(forest_deep_root, newdata = olist_predict)$predictions

pred_final <- (pred_final)^2

#Create CSV-file
Arbian_Halilaj <- data.frame(pred_final)                       
write.csv(Arbian_Halilaj, "/Users/arbiun/Desktop/MECONI/2. Semester/DAI/Individual Paper/Arbian_Halilaj.csv")
