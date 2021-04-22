data <- read.table("/home/kevin/GitRepos/DEN1_GeneralizationAtRetrieval/rsc/freq_data.txt", header=TRUE)


centered <- data.frame(data$Discriminability, data$MedianRT)


centered <- data.frame(scale(data$Discriminability, center=TRUE, scale=FALSE), 
                       scale(data$MedianRT, center=TRUE, scale=FALSE),
                       scale(data$g_fr, center=TRUE, scale=FALSE),
                       scale(data$ug_fr, center=TRUE, scale=FALSE),
                       scale(data$g1_fr, center=TRUE, scale=FALSE),
                       scale(data$g2_fr, center=TRUE, scale=FALSE),
                       scale(data$ug1_fr, center=TRUE, scale=FALSE),
                       scale(data$ug2_fr, center=TRUE, scale=FALSE))

colnames(centered) <- c("d", "RT", "fG", "fUG", "fG1", "fG2", "fUG1", "fUG2")



model1 <- lm(RT ~ d + fG, data=centered)
model2 <- lm(RT ~ d + fUG, data=centered)
model3 <- lm(RT ~ d + fG1, data=centered)
model4 <- lm(RT ~ d + fG2, data=centered)
model5 <- lm(RT ~ d + fUG1, data=centered)
model6 <- lm(RT ~ d + fUG2, data=centered)
model7 <- lm(d ~ fG + fUG + fG1 + fG2 + fUG1 + fUG2, data=centered)


print(summary(model1))
print(summary(model2))
print(summary(model3))
print(summary(model4))
print(summary(model5))
print(summary(model6))
