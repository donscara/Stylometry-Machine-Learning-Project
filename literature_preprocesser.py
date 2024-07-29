import pandas as pd
import inflect
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from textblob import TextBlob


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


file_path = 'datasets/Literature.csv'
df = pd.read_csv(file_path)
main = df.copy()
print(df.head())  


df['Text'] = df['Text'].str.lower()

def remove_punctuation(text):
    chars = [",", ".", "'", '“', '—', '’', '‘', "!", '"', '”', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
             "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    for char in chars:
        text = text.replace(char, '')
    return text

df['Pre-Text'] = df['Text'].apply(remove_punctuation)

p = inflect.engine()

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
df['Text'] = df['Text'].apply(convert_number)

def count_sentences(text):
    sentences = sent_tokenize(text)
    return len(sentences)

df['Sentence-Count'] = df['Text'].apply(count_sentences)
sent_count = df['Sentence-Count'].sum()
print('The total Sentence-Count:', sent_count)

def lexical_diversity(text):
    blob = TextBlob(text)
    num_words = len(blob.words)
    return len(set(blob.words)) / num_words if num_words else 0

df['Lexical-Diversity'] = df['Text'].apply(lexical_diversity)

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

df['Pre-Text'] = df['Pre-Text'].apply(remove_stopwords)
df_ = df.copy()
df_15 = df.drop(columns=['Sentence-Count', 'Lexical-Diversity', 'Pre-Text', 'Text-Length', 'AuthorLastName'])

def create_segments(sequence, segment_size=15, buffer=5):
    sentences = sent_tokenize(sequence)
    num_of_chunks = ((len(sentences) - segment_size) // buffer) + 1
    segments = [" ".join(sentences[i:i + segment_size]) for i in range(0, num_of_chunks * buffer, buffer)]
    return segments

def expand_dataframe(df):
    new_rows = []
    for _, row in df.iterrows():
        segments = create_segments(row['Text'], 10, 10)
        for segment in segments:
            new_row = row.to_dict()
            new_row['Text'] = segment
            new_rows.append(new_row)
    return pd.DataFrame(new_rows)

df_15 = expand_dataframe(df_15)

text_count_by_author = df_15.groupby('Author')['Text'].count()
print('The number of 15-Sent-Segment by Author:', text_count_by_author)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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

csv_file_path = 'datasets/PreprocessedBooksAs15SentenceSegments.csv'
df_15.to_csv(csv_file_path, index=False)

print('Length of df_15:', len(df_15))
