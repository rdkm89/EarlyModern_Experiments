# Last edits: 2019-01-25
# Structural Topic Modelling of Early Modern Drama
# Document already pre-processed in Python
# Cf ~/pre-processing.py

# Libraries
rm(list=ls())
library(stm)
library(igraph)
library(stmCorrViz)

# --------------------------------
# 1. INGEST
# --------------------------------

setwd("~/Desktop/")
# Read data
data <- read.csv("corpus.csv", header=TRUE)


# --------------------------------
# 2. PREPARE
# --------------------------------

# Custom stop word list - expand as necessary!
# Pre-process docs - NO STEMMING!!!
# See http://www.cs.cornell.edu/~xanda/winlp2017.pdf
processed <- textProcessor(data$documents, metadata=data,
                            lowercase=FALSE, removepunctuation=FALSE,
                              stem=FALSE, removestopwords=FALSE)
# Prepare processed docs
max.docs = floor(7105*0.85)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta, 
                      upper.thresh=max.docs)
# Split to separate variables
docs <- out$documents
vocab <- out$vocab
meta <- out$meta
# Show how many tokens + docs would be removed with different thresholds
plotRemoved(processed$documents, lower.thresh=seq(1,200, by=100))
dev.off()

# --------------------------------
# 3. ESTIMATE
# --------------------------------

# Estimate structural topic model with the topic prevalence parameter
## Prevalence of topics varies across document meta data, including 'rating' and 'day'
## 's(day)' applies a spline normalization to 'day' variable.
poliblogPrevFit <- stm(out$documents, out$vocab, K=30, prevalence=~genre+year, 
                        max.em.its=75, data=out$meta, init.type="Spectral", 
                          seed=1234)
# Plot 
pdf("stm-plot-prevfit.pdf", width=10, height=8.5)
plot(poliblogPrevFit)
dev.off()

# Plot ummary for all 20 topics
pdf("stm-plot-prevfit-summary.pdf", width=10, height=8.5)
plot(poliblogPrevFit, type="summary", xlim=c(0,.4))
dev.off()

# Summary for topics 3, 7, and 20
pdf("stm-plot-prevfit-labels.pdf", width=10, height=8.5)
plot(poliblogPrevFit, type="labels", topics=c(3,7,20))
dev.off()

# Histogram of topics
pdf("stm-plot-prevfit-hist.pdf", width=10, height=8.5)
plot(poliblogPrevFit, type="hist")
dev.off()

# Comparing topics 7 and 10
pdf("stm-plot-prevfit-perspectives-two-topic.pdf", width=10, height=8.5)
plot(poliblogPrevFit, type="perspectives", topics=c(4,5))
dev.off()

# --------------------------------
# 4. EVALUATE
# --------------------------------

# Multiple runs - search and select model best model
## [...] assists the user in finding and selecting a model with desirable properties in 
## both semantic coherence and exclusivity dimensions 
## (e.g., models with average scores towards the upper right side of the plot)"
poliblogSelect <- selectModel(out$documents, out$vocab, K=20, prevalence=~rating+s(day),
                                max.em.its=75, data=meta, runs=20, seed=8458159)
# Plot all different models
# Coherence x Exclusivity
pdf("stm-plot-selected.pdf", width=10, height=8.5)
plotModels(poliblogSelect)
dev.off()

# Display topic distribution
# Coherence x Exclusivity
pdf("stm-plot-topic-quality.pdf", width=10, height=8.5)
topicQuality(model=poliblogPrevFit, documents=docs)
dev.off()

# Manually select best model
selectedModel3 <- poliblogSelect$runout[[3]]

# Same thing but with different range of topics
## "For each number of topics, selectModel() is run multiple times. 
## The output is then processed through a function that takes a pareto dominant run 
## of the model in terms of exclusivity and semantic coherence. 
## If multiple runs are candidates (i.e., none weakly dominates the others), 
## a single model run is randomly chosen from the set of undominated runs."
storage <- manyTopics(out$documents, out$vocab, K=c(20:30), prevalence=~genre+year,
                        data=meta, runs=10)

# Plot each model
storageOutput1 <- storage$out[[1]] # For example, choosing the model with 7 topics
pdf("stm-plot-storage-output1.pdf", width=10, height=8.5)
plot(storageOutput1)
dev.off()
storageOutput2 <- storage$out[[2]] # 8 topics
pdf("stm-plot-storage-output2.pdf", width=10, height=8.5)
plot(storageOutput2)
dev.off()
storageOutput3 <- storage$out[[3]] # 9 topics
pdf("stm-plot-storage-output3.pdf", width=10, height=8.5)
plot(storageOutput3)
dev.off()
storageOutput4 <- storage$out[[4]] # 10 topics
pdf("stm-plot-storage-output4.pdf", width=10, height=8.5)
plot(storageOutput4)
dev.off()


# "Alternatively, R can be instructed to figure out the best model 
# automatically defined by exclusivity and semantic coherence for each K"
kResult <- searchK(out$documents, out$vocab, K=c(10,30), prevalence=~rating+s(day),
                    data=meta)
pdf("stm-plot-searchk.pdf", width=10, height=8.5)
plot(kResult)
dev.off()

# --------------------------------
# 5. UNDERSTAND
# --------------------------------

# labelTopics().
# Label topics by listing top words for selected topics 3, 7, 20. Save as txt file.
labelTopicsSel <- labelTopics(poliblogPrevFit, c(3,7,20))
sink("stm-list-label-topics-selected.txt", append=FALSE, split=TRUE)
print(labelTopicsSel)
sink()
# Label topics by listing top words for all topics. Save as txt file.
labelTopicsAll <- labelTopics(poliblogPrevFit, c(1:20))
sink("stm-list-label-topics-all.txt", append=FALSE, split=TRUE)
print(labelTopicsAll)
sink()

# This fucntion can be used as a more detailed alternative to labelTopics()
# Displays verbose labels that describe topics and topic-covariate groups in depth
sink("stm-list-sagelabel.txt", append=FALSE, split=TRUE)
print(sageLabels(poliblogPrevFit))
sink()

# Using findThoughts() function reads documents that are highly correlated with the 
# user-specified topics.
# Here we see n=3 docs to do with topics=3 for first 250 words using texts=shortdoc
thoughts3 <- findThoughts(poliblogPrevFit, texts=shortdoc, n=3, topics=3)$docs[[1]]
plotQuote(thoughts3, width=40, main="Topic 3")
pdf("stm-plot-find-thoughts3.pdf", width=10, height=8.5)
plotQuote(thoughts3, width=40, main="Topic 1")
dev.off()

# The estimateEffect() function explores how prevalence of topics varies across documents 
# according to document covariates (metadata).
out$meta$rating <- as.factor(out$meta$rating)
prep <- estimateEffect(1:20 ~ rating+s(day), poliblogPrevFit, meta=out$meta, 
                        uncertainty="Global")

# To see how prevalence of topics differs across values of a categorical covariate:
# (e.g. political affiliation)
pdf("stm-plot-estimate-effect-categorical.pdf", width=10, height=8.5)
plot(prep, covariate="rating", topics=c(3, 7, 20), model=poliblogPrevFit, 
       method="difference", cov.value1="Liberal", cov.value2="Conservative",
         xlab="More Conservative ... More Liberal", main="Effect of Liberal vs. Conservative",
           xlim=c(-.15,.15), labeltype ="custom", custom.labels=c('Obama', 'Sarah Palin', 
                                                                     'Bush Presidency'))
dev.off()

# To see how prevalence of topics differs across values of a continuous covariate:
# (e.g date)
pdf("stm-plot-estimate-effect-continuous.pdf", width=10, height=8.5)
plot(prep, "day", method="continuous", topics=20, model=z, printlegend=FALSE, xaxt="n", 
       xlab="Time (2008)")
monthseq <- seq(from=as.Date("2008-01-01"), to=as.Date("2008-12-01"), by="month")
monthnames <- months(monthseq)
axis(1, at=as.numeric(monthseq)-min(as.numeric(monthseq)), labels=monthnames)
dev.off()

# topicCorr() an STM permits correlations between topics. Positive correlations between topics 
# indicate that both topics are likely to be discussed within a document. 
mod.out.corr <- topicCorr(poliblogPrevFit)
pdf("stm-plot-topic-correlations.pdf", width=10, height=8.5)
plot(mod.out.corr)
dev.off()

# --------
# 6. Visualise
# --------

# Topical content
# STM can plot the influence of covariates included in as a topical content covariate. 
# A topical content variable allows for the vocabulary used to talk about a particular topic 
# to vary. 
poliblogContent <- stm(out$documents, out$vocab, K=20, prevalence=~rating+s(day), 
                        content=~rating, max.em.its=75, data=out$meta, 
                         init.type="Spectral", seed=8458159)
pdf("stm-plot-content-perspectives.pdf", width=10, height=8.5)
plot(poliblogContent, type="perspectives", topics=7)
dev.off()

# Word cloud
# WORD CLOUD.
pdf("stm-plot-prevfit-wordcloud.pdf", width=10, height=8.5)
cloud(poliblogPrevFit, topic=7)
dev.off()
pdf("stm-plot-content-wordcloud.pdf", width=10, height=8.5)
cloud(poliblogContent, topic=7)
dev.off()

# Covariate interactions
# Interactions between covariates can be examined such that one variable may “moderate” the 
# effect of another variable.
poliblogInteraction <- stm(out$documents, out$vocab, K=20, prevalence=~rating*day, 
                            max.em.its=75, data=out$meta, seed=8458159)

# Prep covariates using the estimateEffect() function, only this time, we include the 
# interaction variable.
prep2 <- estimateEffect(c(20) ~ rating*day, poliblogInteraction, metadata=out$meta, 
                         uncertainty="None")
pdf("stm-plot-interact-estimate-effect.pdf", width=10, height=8.5)
plot(prep2, covariate="day", model=poliblogInteraction, method="continuous", xlab="Days",
       moderator="rating", moderator.value="Liberal", linecol="blue", ylim=c(0,0.12), 
         printlegend=F)
plot(prep2, covariate="day", model=poliblogInteraction, method="continuous", xlab="Days",
       moderator="rating", moderator.value="Conservative", linecol="red", add=T,
        printlegend=F)
legend(0,0.12, c("Liberal", "Conservative"), lwd=2, col=c("blue", "red"))
dev.off()

# Plot convergence
pdf("stm-plot-prevfit-convergence.pdf", width=10, height=8.5)
plot(poliblogPrevFit$convergence$bound, type="l", ylab="Approximate Objective", 
      main="Convergence")
dev.off()


# stmCorrViz() function for the package of the same name generates an interactive 
# visualisation of topic hierarchy/correlations in a structural topicl model.
# Export to JSON; then D3.
stmCorrViz(poliblogPrevFit, "stm-interactive-correlation.html", 
            documents_raw=data$documents, documents_matrix=out$documents)


