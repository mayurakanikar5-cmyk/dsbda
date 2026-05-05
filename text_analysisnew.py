import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

text = """Data science is an interdisciplinary field that uses scientific methods,
processes, algorithms and systems to extract knowledge and insights from data."""
print(text)


import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print(tokens)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
print("\nAfter Stopword Removal:")
print(filtered_tokens)

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

pos_tags = nltk.pos_tag(filtered_tokens)
print(pos_tags)

from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed = [ps.stem(w) for w in filtered_tokens]
print("\nStemmed Words:")
print(stemmed)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered_tokens]
print("\nLemmatized Words:")
print(lemmatized)

from collections import Counter
tf = Counter(filtered_tokens)
print("\nTerm Frequency:")
print(tf)

from sklearn.feature_extraction.text import TfidfVectorizer
documents = [text]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
print("\nTF-IDF Values:")
print(tfidf_matrix.toarray())
print("\nFeature Names:")
print(vectorizer.get_feature_names_out())