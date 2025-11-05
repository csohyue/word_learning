library(pso)
library(tidyverse)

# Associative Uncertainty- (Entropy) & Familiarity-Biased Model
# George Kachergis  george.kachergis@gmail.com

## Some edits by Christine Yue csohyue@upenn.edu

shannon.entropy <- function(p) {
  if (min(p) < 0 || sum(p) <= 0)
    return(NA)
  p.norm <- p[p>0]/sum(p)
  -sum(log2(p.norm)*p.norm)
}

update_known <- function(m, tr_w, tr_o) {
  startval = .01
  
  for(i in tr_w) { # for each word in the input
    for(c in 1:dim(m)[2]) { # for each column
      if(sum(m[,c]>0) & m[i,c]==0) { 
        # if the sum of the column is greater than 0 (i.e. the object has previously
        # been seen) and the word-object pair is 0, set it to the startval
        m[i,c] = startval
        # cross-initialization
        # if(c<nrow(m)) m[c,i] = startval # Something weird is going on here
      }
    }
    for(j in tr_o) { # for each object
      if(m[i,j]==0) m[i,j] = startval # set the word, object to startval if 0
      # cross-initialization
      # if(j<nrow(m) && m[j,i]==0) m[j,i] = startval # Something weird is going on here
    }
  }
  return(m)
}

get_perf <- function(m) {
  perf <- rep(0, nrow(m))
  names(perf) <- rownames(m)
  for (ref in colnames(m)) {
    if (!(ref %in% rownames(m))) {
      next
    }
    correct <- m[ref, ref]
    total <- sum(m[ref,])
    if (total == 0) {
      next
    }
    perf[ref] <- correct / total
  }
  return(perf)
}

model <- function(params, ord, reps=1, test_noise=0) {
  X <- params[1] # associative weight to distribute
  B <- params[2] # weighting of uncertainty vs. familiarity
  C <- params[3] # decay
  
  voc_sz = max(unlist(ord$words), na.rm=TRUE) # vocabulary size
  ref_sz = max(unlist(ord$objs), na.rm=TRUE) # number of objects
  
  traj = list()
  m <- matrix(0, voc_sz, ref_sz) # association matrix
  perf = matrix(0, reps, voc_sz) # a row for each block
  # training
  for(rep in 1:reps) { # for trajectory experiments, train multiple times
    for(t in 1:nrow(ord$words)) {
      
      tr_w = as.integer(ord$words[t,])
      tr_w = tr_w[!is.na(tr_w)]
      tr_o = as.integer(ord$objs[t,])
      tr_o = tr_o[!is.na(tr_o)]
      m = update_known(m, tr_w, tr_o) # what's been seen so far?
      ent_w = c() # more entropy = more dispersive
      for(w in tr_w) { ent_w = c(ent_w, shannon.entropy(m[w,])) }
      ent_w = exp(B*ent_w)
      
      ent_o = c() # more entropy = more dispersive
      for(o in tr_o) { ent_o = c(ent_o, shannon.entropy(m[,o])) }
      ent_o = exp(B*ent_o)
      
      nent = (ent_w %*% t(ent_o))
      assocs = m[tr_w,tr_o]
      denom = sum(assocs * nent)
      m = m*C # decay everything
      # update associations on this trial
      m[tr_w,tr_o] = m[tr_w,tr_o] + (X * assocs * (ent_w %*% t(ent_o))) / denom
      
      index = (rep-1)*length(ord$trials) + t # index for learning trajectory
      traj[[index]] = m
    }
    m_test = m+test_noise # test noise constant k
    perf[rep,] = diag(m_test) / rowSums(m_test)
  }
  want = list(perf=perf, matrix=m, traj=traj)
  return(want)
}

run_model <- function(cond, parameters, print_perf=F) {
  require(pso) # or require(DEoptim)
  mod = model(parameters, ord=cond$train)
  if(print_perf) print(mod$perf)
  return(mod)
}

meanSSE <- function(par, order, human_perf) {
  mod = model(par, order$train)
  return(sum((mod$perf-human_perf)^2))
}

fit_model <- function(model_name, order, par_lower, par_upper, se_fn) {
  lowest_SSE = 100
  best_par = NA
  par_init = (par_lower + par_upper) / 2
  startt = Sys.time()
  cat("Model\tOrder\tSSE\tParameters\n")
  for(test_row in 1:20) { 
    best <- psoptim(par_init, se_fn, ord=order, human_perf=unlist(order$HumanItemAcc), lower=par_lower, upper=par_upper)
    if(best$value < lowest_SSE){
      best_par = best$par
      lowest_SSE = best$value
    }
    cat(model_name,'\t',order$Condition,'\t',best$value,'\t',best$par,'\n')
  }
  stopt = Sys.time()
  print(stopt-startt)
  print(paste0(best_par, collapse=", "))
  return(best_par)
}

# Christine defined this function: Kachergis approximates test over the entire 
# association matrix, but in CSWL studies, there are typically fewer foils than 
# the entire set. In the case where we know the test, we define the test 
# function. Otherwise, we use the approximation of perf.
test_model <- function(expt, params, test_words, test_objects, exposure_length) {
  mod = run_model(expt, params, print_perf=F)
  final_trajectory = mod$traj[exposure_length][[1]]
  
  test_performance = matrix(0, 1, nrow(test_words))
  for(test_row in 1:nrow(test_words)) {
    test_options = as.integer(test_objects[test_row,])
    # The first item in the option list must be the target referent
    numerator = final_trajectory[test_words[test_row,], test_options[1]]
    denominator = 0
    for(option in test_options) {
      denominator = denominator + final_trajectory[test_words[test_row,], option]
    }
    test_performance[1,test_row] = numerator / denominator
  }
  
  return(test_performance)
}
