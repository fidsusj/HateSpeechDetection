# Research results

## I. Hate Speech on Twitter: A Pragmatic Approach to Collect Hateful and Offensive Expressions and Perform Hate Speech Detection

- Training set: 7.000 tweets per class (hate speech, offensive language, neither)
- Test set: 670 tweets per class
- Validation set: 670 tweets per class
- Preprocessing: removal of tags, URLs and irrelevant expressions, tokenization, PoS tagging, lemmatization
- Feature extraction: sentiment polarity, all-capitalized words, set of words (unigrams), expressions (patterns)

To conclude, mainly 4 sets of features are extracted which
we qualify as ‘‘sentiment-based features‘‘, ‘‘semantic features,’’ ‘‘unigram features‘‘, and ‘‘pattern features.’’ By combining these sets, we believe it is possible to detect hate
speech: ‘‘sentiment features‘‘ allow us to extract the polarity
of the tweet, a very essential component of hate speech (given
that hateful speeches are mostly negative ones). ‘‘Semantic features’’ allow us to find any emphasized expression.
‘‘Unigram features’’ allow us to detect any explicit form of
hate speech, whereas patterns allow the identification of any
longer or implicit forms of hate speech.

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

## For our work

- Define hate speech and offensive language
- Related work
- Define Framework for hate speech detection
- Introduce own dictionary of unigrams and patterns
- Define dataset
- Define features
- Define classifiers
- Evaluation
