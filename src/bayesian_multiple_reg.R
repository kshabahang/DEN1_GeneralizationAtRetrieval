library(rstan)
library(coda)
library(blme)


data <- read.table("/home/hal9000/Repos/DEN1_GeneralizationAtRetrieval/rsc/freq_data.txt", header=TRUE)


centered <- data.frame(scale(data$Discriminability, center=TRUE, scale=FALSE), 
                       scale(data$MedianRT, center=TRUE, scale=FALSE),
                       scale(data$g_fr, center=TRUE, scale=FALSE),
                       scale(data$ug_fr, center=TRUE, scale=FALSE),
                       scale(data$g1_fr, center=TRUE, scale=FALSE),
                       scale(data$g2_fr, center=TRUE, scale=FALSE),
                       scale(data$ug1_fr, center=TRUE, scale=FALSE),
                       scale(data$ug2_fr, center=TRUE, scale=FALSE))

colnames(centered) <- c("d", "RT", "fG", "fUG", "fG1", "fG2", "fUG1", "fUG2")


fileName <- "./mult_reg.stan"
model <- readChar(fileName, file.info(fileName)$size)

dat <- list(N = nrow(centered), 
            K = 8,
            RT = centered$RT,
            d = centered$d,
            fG = centered$fG, 
            fUG = centered$fUG, 
            fG1 = centered$fG1, 
            fG2 = centered$fG2, 
            fUG1 = centered$fUG1, 
            fUG2 = centered$fUG2)

res <- stan(fileName, data=dat, chains=3, iter=10000, warmup=1000, thin=10)

beta0 <- extract(res,pars=c("beta[1]"))
beta1 <-extract(res,pars=c("beta[2]"))
beta2 <-extract(res,pars=c("beta[3]"))
beta3 <-extract(res,pars=c("beta[4]"))
beta4 <-extract(res,pars=c("beta[5]"))
beta5 <-extract(res,pars=c("beta[6]"))
beta6 <-extract(res,pars=c("beta[7]"))
beta7 <-extract(res,pars=c("beta[8]"))

hpd0 <-HPDinterval(as.mcmc(unlist(beta0)),prob=0.95)
hpd1 <-HPDinterval(as.mcmc(unlist(beta1)),prob=0.95)
hpd2 <-HPDinterval(as.mcmc(unlist(beta2)),prob=0.95)
hpd3 <-HPDinterval(as.mcmc(unlist(beta3)),prob=0.95)
hpd4 <-HPDinterval(as.mcmc(unlist(beta4)),prob=0.95)
hpd5 <-HPDinterval(as.mcmc(unlist(beta5)),prob=0.95)
hpd6 <-HPDinterval(as.mcmc(unlist(beta6)),prob=0.95)
hpd7 <-HPDinterval(as.mcmc(unlist(beta7)),prob=0.95)









