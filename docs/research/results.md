# Research results

## I. Hate Speech on Twitter: A Pragmatic Approach to Collect Hateful and Offensive Expressions and Perform Hate Speech Detection

- Training set: 7.000 tweets per class (hate speech, offensive language, neither)
- Test set: 670 tweets per class
- Validation set: 670 tweets per class
- Preprocessing: removal of tags, URLs and irrelevant expressions, tokenization, PoS tagging, lemmatization
- Feature extraction: sentiment polarity, all-capitalized words, set of words (unigrams), expressions (patterns)

To conclude, mainly 4 sets of features are extracted which
we qualify as ‘‘sentiment-based features‘‘, ‘‘semantic features,’’ ‘‘unigram features‘‘, and ‘‘pattern features.’’ By 
combining these sets, we believe it is possible to detect hate speech: ‘‘sentiment features‘‘ allow us to extract the 
polarity of the tweet, a very essential component of hate speech (given that hateful speeches are mostly negative ones). 
‘‘Semantic features’’ allow us to find any emphasized expression. ‘‘Unigram features’’ allow us to detect any explicit 
form of hate speech, whereas patterns allow the identification of any longer or implicit forms of hate speech.

- Sentiment-based features
  - the total score of positive words (PW),
  - the total score of negative words (NW),
  - the ratio of emotional (positive and negative) words ρ(t) (see formula in paper)
  - the number of positive slang words,
  - the number of negative slang words,
  - the number of positive emoticons,
  - the number of negative emoticons,
  - the number of positive hashtags,
  - the number of negative hashtags.
    
The tool used is called _SentiStrength_.

- Semantic features
  - the number of exclamation marks,
  - the number of question marks,
  - the number of full stop marks,
  - the number of all-capitalized words,
  - the number of quotes,
  - the number of interjections,
  - the number of laughing expressions,
  - the number of words in the tweet.
    
- Unigram features

Each of the resulting words will be used as a unique feature: for a word w, in each tweet, we check whether it is 
employed or not. If the tweet contains the word, the value of the corresponding feature is set to ‘‘true’’ otherwise, it 
is set to ‘‘false.’’

- Pattern features:

A pattern is extracted from a tweet as follows: for each word, if it belongs to ‘‘SW,’’ it is replaced by its simplified
PoS tag as described in TABLE 1 along with its polarity. For example the word ’’coward’’ will be replaced by the
expression ‘‘Negative_ADJECTIVE.’’ Otherwise, if the word belongs to ‘‘NSW’’ it is simply replaced by its simplified PoS
tag as described in TABLE 1. The resulting vectors extracted from different tweets have different lengths, therefore, we 
define a pattern as a vector of consecutive words having a fixed length L where L is a parameter to optimize. If a 
tweets has more than L words, we extract all possible patterns. If it has less words than L, it is simply discarded.

Using the optimal values of the two parameters minpocc and Thp, 1875 patterns features are extracted in total.

- Classification algorithms
  - Random Forest
  - SVM
  - J48graft
    
- Evaluation:

Our proposed approach reaches an accuracy equal to 87.4% for the binary classification of tweets into offensive and 
non-offensive, and an accuracy equal to 78.4% for the ternary classification of tweets into, hateful, offensive and clean.
In a future work, we will try to build a richer dictionary of hate speech patterns that can be used, along with a 
unigram dictionary, to detect hateful and offensive online texts.

## II. Evaluating Machine Learning Techniques for Detecting Offensive and Hate Speech in South African Tweets

- 15.702 tweets (11.172 train, 3.724 test)
- preprocessing: TweetTokenizer from NLTK, stemming, username/punctuation/special character/hashtags/stop word removal, 
all words to lowercase
  
- word n-gram features per tweet (unigrams and bigrams) weighted with TF-IDF score
- character n-gram features per tweet (trigrams and four-grams) weighted with TF-IDF score
- syntactic based features
  - capital letters such as A, B, . . . , Z
  - small letters such as a, b, . . . , z
  - uppercase words such as COME, RIGHT
  - lowercase words such as dog, land
  - length of tweets including spaces
  - alphanumeric words such as Red, black, White
  - exclamation marks such as !
  - question marks such as ?
  - full stops such as.
  - quotes such as ‘‘’’, ‘’
  - special characters such as @, #, $, %, ^, &, ∗, _,etc
  - hash tags marked with characters such as #Orania
- negative sentiment-based features (+ negative polarity scores)
  - negations
  - negative words based on Opinion Lexicon[28]
  - negative emoticons based on Urban Dictionary11
  - negative emojis based on Twitter based on Urban Dictionary12
  - Hatebase slur words
    
- Classifiers
  - Logistic Regression
  - SVM
  - Random Forest (RF)
  - Gradient Boosting (GB)
  - Ensemble models of these three
  - Multi-tier meta-learning model

Class imbalances were removed by applying synthetic minority oversampling technique (SMOTE).
 
- Evaluation

The results in Table 5 showed that both word n-gram and character n-gram features had the best performances among the 
features. The optimized model of GB with word n-gram technique recorded the best TPR of 0.867 for detection of offensive 
speech, while the SVM with char n-gram technique had the best TPR of 0.894 for detection of hate speech. The results in 
Table 6 for precision, recall and F1 also showed that word and character n-gram had the best performance.

## III. A Survey on Automatic Detection of Hate Speech in Text

- “generic text mining features” and “specific hate speech detection features”
- Generic text mining features (see Fig. 8):
  - Dictionaries: Gather term frequencies and use them as features (insults, swear words, reaction words (https://www.noswearing.com/)
    profane words, verbal abuse, stereotypical utterances, words with negative connotations)
  - Distance Metric: sh1t -> shit: distance=1
  - Bag of Words: term frequencies, but may lead to misclassification
  - N-grams: most used techniques in hate speech detection (also n-grams with characters or syllables => not so 
    susceptible to spelling variations). Higher N values perform better than lower N values. N-gram features perform better
    when combined with other features
  - Profanity Windows: mixture of dictionary approach and n-grams
  - TF-IDF
  - Part-of-speech
  - Lexical Syntactic Feature-based (LSF): capture grammatical dependencies within a sentence
  - Rule based approaches
  - Participant-vocabulary Consistency: tendency of each user to harass ot to be harassed
  - Template Based Strategy: gather k words that appear around a specific word and build a template
  - Word Sense Disambiguation Techniques
  - Typed Dependencies: description of the grammatical relationships in a sentence => extract theme based grammatical 
  patterns
  - Topic classification (Latent Dirichlet Allocation (LDA))
  - Sentiment
  - Word Embeddings: paragraph2vec, FastText. Problem: sentences must be classified and not words => average all word embeddings
  - Deep Learning
  - Named entity recognition, word sense disambiguation techniques to check polarity, frequencies of personal pronouns, 
  emoticons, capital letters, URLs, hashtags, mentions, retweets, number of tags, terms used in the tags, number of notes 
    (reblog and like count), 
- Specific hate speech detection features:
  - Othering Language: "us vs. them"
  - Perpetrator characteristics
  - Objectivity-Subjectivity of the language: hate speech is related with more subjective communication
  - declarations of superiority of the ingroup
  - focus on particular stereotypes
  - intersectionism of oppression
- majority of papers uses between 1.000 and 10.000 labeled instances
- Paper contains a list of datasets! (In case our data is insufficient)
- Most common algorithms used are SVM, Random Forest and Decision Trees, Logistic Regression, Naive Bayes
- Most common metrics: Precision, Recall, F-measure, Accuracy and AUC
- Best results were achieved when deep learning was used

## IV. Detecting Hate Speech and Offensive Language on Twitter using Machine Learning: An N-gram and TFIDF based Approach

- Classes: hateful, offensive and clean
- Data: combination of three datasets, no metrics given
- TF-IDF weighted n-grams as features (Related Work: have a look at Stanford Natural Language Processing Group)
- L1 and L2 normalization
- Logistic Regression (best), SVM and Naive Bayes
- Lowercase and removal of: space patterns, URLs, twitter mentions, retweet symbols, stopwords
- N-gram range 1-3, L2 normalization, c=100 => 95.6% accuracy

## V. Detecting Hate Speech in Social Media

- Hate (2.399), Offensive (4.836) and OK (7.274) as classes
- LIBLINEAR SVM
- Features:
  - n-grams: character n-grams (2-8) (across word boundaries), word n-grams (1-3) (similar to syntactic dependencies)
  - 1-/2-/3-skip word bigrams
- Character 4-grams perform best (accuracy of 78.0%)
- Performance increases with #training instances

## VI. Automated Hate Speech Detection and the Problem of Offensive Language

- 25k tweets (5% hate speech, 77% offensive language, 18% non-offensive)
- Features:
  - TF-IDF weighted unigrams, bigrams and trigrams
  - POS tag unigrams, bigrams and trigrams for syntactic structure
  - To capture the quality of each tweet: modified Flesch-Kincaid Grade Level and Flesch Reading Ease scores, where the 
    number of sentences is fixed at one
  - Sentiment scores based on sentiment lexicon
  - count indicators for hashtags, mentions, retweets and URLs
  - number of characters, words and syllables in each tweet
- Model: 
  - Logistic regression with L1 regularization to reduce the dimensions
  - Logistic regression, naive bayes, decision trees, random forests and linear SVMs. 
- Logistic regression and linear SVM perform significantly better than the other models
- One vs. rest framework => seperate logistic regression classifier with L2 normalization for every class
- 0.91 overall precision, 0.9 recall and 0.9 F1-score, but model is biased towards classifying tweets as less hateful or
offensive than the human coders

## For our work

- Motivation of HateSpeechDetection: See paper III.3
- Define hate speech and offensive language See (See III.4)
- Related work
- Define Framework for hate speech detection
- Introduce own dictionary of unigrams and patterns
- Define dataset
- Define features
- Define classifiers
- Evaluation
