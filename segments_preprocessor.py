import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import nltk
import textstat
import textblob
import scipy.stats
from collections import Counter
import math
import string
from textstat.textstat import textstatistics
import collections as coll
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
import inflect
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

file_path = 'datasets/PreprocessedLiteratureAs15SentenceSegments.csv'
df_ = pd.read_csv(file_path)
df_ = df_.dropna().reset_index(drop=True)

columns = [
    "Book", "Author", "Genre", "Publish-Year", "Average-Rating", "AuthorCode",
    "AverageWordLength", "AverageSentenceLengthByWord", "AverageSyllablePerWord", "SpecialCharactersCount",
    "PunctuationCount", "FunctionalWordsCount", "TypeTokenRatio", "HonoreMeasureR",
    "Hapax", "SichelesMeasureS", "Dihapax", "YulesCharacteristicK", "SimpsonsIndex",
    "BrunetsMeasureW", "ShannonEntropy", "FleschReadingEase", "FleschKincaidGradeLevel",
    "DaleChallReadability", "GunningFog", "AverageSentenceLengthByChar", "Sentiment"
]

df = pd.DataFrame(columns=columns)
p = inflect.engine()


def tokenize_words(text):
    return nltk.word_tokenize(text)

def tokenize_sents(text):
    return nltk.sent_tokenize(text)

def remove_special_chars_and_tokenize(text):
    special_chars = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    words = [word for word in text if word not in special_chars]
    return words

def flesch_reading(text):
    return textstatistics().flesch_reading_ease(text)

def grade_level(text):
    return textstatistics().flesch_kincaid_grade(text)

def gunning_fog(text):
    return textstatistics().gunning_fog_index(text)

def average_word_length(text):
    words = tokenize_words(text)
    total_characters = sum(len(word) for word in words)
    return total_characters / len(words)

def average_sentence_length_by_char(text):
    sentences = tokenize_sents(text)
    return np.average([len(sentence) for sentence in sentences]) if sentences else 0

def average_sentence_length_by_word(text):
    sentences = tokenize_sents(text)
    words = tokenize_words(text)
    return len(words) / len(sentences)

def average_syllables_per_word(text):
    words = tokenize_words(text)
    return textstat.syllable_count(text) / len(words)

def count_special_characters(text):
    special_characters = set("#$%&()*+-/<=>@[\\]^_`{|}~\t\n")
    return sum(char in special_characters for char in text) / max(1, len(text))

def count_punctuation(text):
    chars = set(",.'!\";?:")
    return sum(char in chars for char in text) / max(1, len(text))

def count_functional_words(text):
    words = tokenize_words(text)
    functional_words = nltk.corpus.stopwords.words('english')
    num_functional_words = sum(1 for word in words if word.lower() in functional_words)
    return num_functional_words / len(words)

def dale_chall_readability(text):
    return textstat.dale_chall_readability_score(text)

def simpsons_index(text):
    words = tokenize_words(text)
    word_freq = Counter(words)
    return 1 - sum((freq / len(words)) ** 2 for freq in word_freq.values())

def shannon_entropy(text):
    words = tokenize_words(text)
    word_freq = Counter(words)
    probs = [freq / len(words) for freq in word_freq.values()]
    return scipy.stats.entropy(probs, base=2)

def yules_characteristic_k(text):
    words = tokenize_words(text)
    N = len(words)
    word_freq = Counter(words)
    M1 = N
    M2 = sum(freq * (freq - 1) for freq in word_freq.values())
    return 10**4 * ((M2 / (M1**2)) + (1 / M1))

def brunets_measure_w(text):
    words = tokenize_words(text)
    V = len(set(words))
    N = len(words)
    return N * (V ** -0.172)

def type_token_ratio(text):
    words = tokenize_words(text)
    V = len(set(words))
    N = len(words)
    return V / N

def hapax_dis_legomena(text):
    words = tokenize_words(text)
    word_freq = Counter(words)
    h = sum(1 for _, freq in word_freq.items() if freq == 2)
    s = 2 * h / len([word for word, count in word_freq.items() if count == 1])
    return s, h

def hapax_legomena(text):
    words = tokenize_words(text)
    N = len(words)
    h = len([word for word, count in Counter(words).items() if count == 1]) / len(words)
    honore_r = (100 * math.log(N)) / (1 - h)
    return honore_r, h

def sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

for index, row in df_.iterrows():
    text = row['Text']
    honore_measure_r, hapax = hapax_legomena(text)
    sicheles_measure_s, dihapax = hapax_dis_legomena(text)
    new_entry = {
        "Book": row['Book'],
        "Text": row['Text'],
        "Author": row['Author'],
        "Genre": row['Genre'],
        "Publish-Year": row['Publish-Year'],
        "Average-Rating": row['Average-Rating'],
        "AuthorCode": row['AuthorCode'],
        "AverageWordLength": average_word_length(text),
        "AverageSentenceLengthByWord": average_sentence_length_by_word(text),
        "AverageSentenceLengthByChar": average_sentence_length_by_char(text),
        "AverageSyllablePerWord": average_syllables_per_word(text),
        "SpecialCharactersCount": count_special_characters(text),
        "PunctuationCount": count_punctuation(text),
        "FunctionalWordsCount": count_functional_words(text),
        "TypeTokenRatio": type_token_ratio(text),
        "HonoreMeasureR": honore_measure_r,
        "Hapax": hapax,
        "SichelesMeasureS": sicheles_measure_s,
        "Dihapax": dihapax,
        "YulesCharacteristicK": yules_characteristic_k(text),
        "SimpsonsIndex": simpsons_index(text),
        "BrunetsMeasureW": brunets_measure_w(text),
        "ShannonEntropy": shannon_entropy(text),
        "FleschReadingEase": flesch_reading(text),
        "FleschKincaidGradeLevel": grade_level(text),
        "DaleChallReadability": dale_chall_readability(text),
        "GunningFog": gunning_fog(text),
        "Sentiment": sentiment(text)
    }
    new_row = pd.DataFrame([new_entry])
    df = pd.concat([df, new_row], ignore_index=True)

df['Pre-Text'] = df['Text'].str.lower()

def remove_punctuation(text):
    chars = [",", ".", "'", '“', '—', '’', '‘', "!", '"', '”', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
             "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    for char in chars:
        text = text.replace(char, '')
    return text

df['Pre-Text'] = df['Pre-Text'].apply(remove_punctuation)

def convert_number(text):
    temp_str = text.split()
    new_string = []
    for word in temp_str:
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)
        else:
            new_string.append(word)
    return ' '.join(new_string)

df['Pre-Text'] = df['Pre-Text'].apply(convert_number)

def remove_stopwords(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

df['Pre-Text'] = df['Pre-Text'].apply(remove_stopwords)

lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_text(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    nltk_pos_tagged = pos_tag(filtered_words)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_pos_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(lemmatizer.lemmatize(word))
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

df['Lemmatized-Text'] = df['Pre-Text'].apply(lemmatize_text)


print('Length of the dataframe is:', str(len(df)))
print(f"Does the DataFrame have any NaN values? {df.isna().sum().sum()}")
has_nan = df.isnull().values.any()
print(f"Are there any NaN values in the DataFrame? {has_nan}")
df = df.dropna().reset_index(drop=True)
print(f"Does the DataFrame have any NaN values? {df.isna().sum().sum()}")
has_nan = df.isnull().values.any()
print(f"Are there any NaN values in the DataFrame? {has_nan}")

csv_file_path = 'datasets/Vectorized15SentenceSegments.csv'
df.to_csv(csv_file_path, index=False)
