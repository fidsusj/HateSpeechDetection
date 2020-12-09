# Milestone - HateSpeechDetection

- Christopher Klammt (iv249@stud.uni-heidelberg.de)
- Felix Hausberger (eb260@stud.uni-heidelberg.de)
- Nils Krehl (pu268@stud.uni-heidelberg.de)

## Project State

[ToDo]
- Planning State
- Future Planning
- High-level Architecture Description
- Experiments (baselines)

## Data Analysis

### Data Sources

As mentioned in the project proposal we are using two data sets for this project:
- [Automated Hate Speech Detection and the Problem of Offensive Language](https://github.com/t-davidson/hate-speech-and-offensive-language) uses data from the Twitter API and classifies the round about 25k tweets as hate speech, offensive language or neutral (ternary classification style)
- [Hate speech dataset from a white supremacist forum](https://github.com/Vicomtech/hate-speech-dataset) consists of data extracted from the White Supremacy Forum and classifies the sentences in a binary fashion as hate or no hate

### Preprocessing

Due to the differences in the data sets - mainly one being binary labeled and the other being ternary labeled - the two data sets first have to be unified and merged into one data set.
To do so, the function `_prepare_and_merge_datasets()` calls the specific function for the separate data sets.
The data from the Twitter API is preprocessed, such that all documents containing offensive language are dropped, as for now we want to concentrate on classifying hate speech.
Both data sets are merged into one while standardizing them, so that all sentences containing hate speech are labeled with a `0` and the neutral samples are labeled with a `1`.

After merging the two data sets, the actual preprocessing can start (executed in the `build_corpus` function in the `corpus.py` file).
First of all the sentences are converted to lower case and all none-word characters are removed.
In the second step, the sentences are tokenized, such that every datapoint consists of a list of tokens (=words).
Furthermore, the tokens are cleaned, meaning that single characters, i.e. all tokens shorter than 3 characters are omitted.
Based on these cleaned tokens, stemming is conducted. This simplifies the phrases by removing endings (as discussed in the course).

![Preprocessing pipeline example](figures/preprocessing-pipeline-example.png)

In the image above this pipeline is illustrated by means of an example sentence.

### Basic Statistics

The following image shows the distribution of the data points.

![Data distribution](figures/distribution.png)

As one can see, there are more sentences in the combined data set marked as neutral than labeled as containing hate speech.
There are 13,335 neutral sentences and 2,490 hate speech examples in our merged data set.

As a comparison one can see the bar charts of the 15 most common tokens for hate speech vs. non-hate speech.
These do not really differ, because most of the words are stop words which are to be removed in the processing pipeline.

![Bar Chart - Hate Speech - all tokens](figures/BarChart-HateSpeech-alltokens.png)
![Bar Chart - Non-Hate Speech - all tokens](figures/BarChart-Non-HateSpeech-alltokens.png)


The differences in hate speech and non-hate speech get more clear, when looking at the cleaned tokens.
This is illustrated in the following wordclouds, which are based on the 15 most common cleaned tokens for each class respectively.

![Wordcloud - Hate Speech - cleaned tokens](figures/Wordcloud-HateSpeech-cleanedtokens.png)
![Wordcloud - Non-Hate Speech - cleaned tokens](figures/Wordcloud-Non-HateSpeech-cleanedtokens.png)

The word clouds are based on the following bar chart distributions:

![Bar Chart - Hate Speech - cleaned tokens](figures/BarChart-HateSpeech-cleanedtokens.png)
![Bar Chart - Non-Hate Speech - cleaned tokens](figures/BarChart-Non-HateSpeech-cleanedtokens.png)

See the folder `figures` for more illustration or the jupyter notebook `data_visualization.ipynb` for the complete code.

### Examples

For a better insight into the data set, a few examples are shown in the following.

Examples for non-hate speech:
- "billy that guy would nt leave me alone so i gave him the trudeau salute"
- "this is after a famous incident of former prime minister pierre trudeau who gave the finger to a group of protesters who were yelling antifrench sayings at him"
- "askdems arent you embarrassed that charlie rangel remains in your caucus"

These are neutral sentences, including a rather incomprehensible example (the last sentence).

Examples for hate speech:
- "california is full of white trash"
- "and yes they will steal anything from whites because they think whites owe them something so it s ok to steal"
- "why white people used to say that sex was a sin used to be a mystery to me until i saw the children of browns and mixed race children popping up all around me"

One can clearly see the hate expressed in the hate speech examples and see their discriminating nature.
