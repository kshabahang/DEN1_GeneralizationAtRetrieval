data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  real RT[N];      // outcome vector
  real d[N]; //discriminability
  real fG[N]; //grammatical bi-gram frequency
  real fUG[N]; //ungrammatical bi-gram frequency
  real fG1[N]; //grammatical first constituent frequency
  real fG2[N]; //grammatical second constituent frequency
  real fUG1[N]; //ungrammatical first constituent frequency
  real fUG2[N]; //ungrammatical second constituent frequency
}
parameters {
  vector[K] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale
}
model {
  for (i in 1:N) {
    //flat prior
    RT[i] ~ normal(beta[1] + beta[2]*d[i] + beta[3]*fG[i] + beta[4]*fUG[i] + beta[5]*fG1[i] + beta[6]*fG2[i] + beta[7]*fUG1[i]+ beta[8]*fUG2[i], sigma);  // likelihood
    }
}
