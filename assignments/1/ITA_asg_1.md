# Assignment 1

## Problem 1-1 Text Analytics Pipeline

### 1.

#### (i)

The driver for this problem could either be text-driven or problem-driven. But we probably assume, that the user feedback contains complaints regarding things they don't like as well as positive feedback. So it rather falls into the problem-driven space, because we have the background theory, that the data contains helpful feedback for the further development and we try to summarize this. Which is a clear goal we would like to achieve.

#### (ii)

This problem clearly is method-driven, as we found out about a new method and want to implement it into our current pipeline.

### 2.

Advantages:
- one can use different sampling rates for the different subgroups, if a certain subgroup is deemed as more important (for example a certain age or income group)
- in overall it should reduce the sampling error
- interesting to have results for the different subgroups as well

Disadvantages:
- is more time- and resource-consuming, because of the extra overhead of forming the subgroups
- not straight-forward to apply it

In general when doing text-driven analysis, it is not that suitable to use stratified sampling, as we have to have some insights and hypothesis about the data and how to form subgroups. Furthermore, the subgroups should be at least somewhat of the same size or contain a minimum of data points. Otherwise a subgroup may contain just one or a few examples which would lead to these definitely being in the stratified sample.

### 3. 

#### (i) Stop word removal

Advantages:
- saves space in database
- saves valuable processing time
- often are not semantically relevant ("Peter and Susan" -> the "and" does not contain any additional meaning)

Disadvantages:
- problems arise when searching for phrases that only or mainly contain stop words (e.g. "Take That" or "The Who")
- in some cases the stop words can actually change the whole meaning of a sentence (e.g. "report as the CEO" vs. "report to the CEO")

#### (ii) Stemming

Advantages:
- simplifies the phrases and the total amount of words necessary that need to be understood (stem "development" and "developer" to "develop", while maintaining the general meaning)
- again saves space in database and accelerates processing

Disadvantages:
- removes semantic information to some extent, as different forms of words contain meaning as well (e.g. "developing" is currently in action, whereas "development" means a process)
- chopping of plural endings also removes sometimes very important information

## Problem 1-2 PDF Conversion and Regular Expressions
