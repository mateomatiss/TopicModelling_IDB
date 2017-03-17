'''
Created on Mar 7, 2017

@author: Mateo
'''
import os
from time import time
import regex
import nltk
import itertools
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.decomposition import NMF, LatentDirichletAllocation
from html import parser  # @UnresolvedImport pydev bug
from wordcloud import WordCloud  # @UnresolvedImport
#from nltk.corpus import stopwords

start_time = time()
path_neg = os.path.dirname(os.path.abspath(__file__)) + '\\neg'
path_pos = os.path.dirname(os.path.abspath(__file__)) + '\\pos'


APOSTROPHES = {"'s" : "is", "'re" : "are", "Im" : "I am", "'ve" : "have",  "'m"  : "am", "'d"  : "would", "'t"  : "not", "'ll" : "will", "Ive" : "I have", "isn" : "is", "aren" : "are", "wasn" : "was", "weren" : "were", "haven" : "have", "hasn" : "has", "hadn" : "had", "won" : "will", "wouldn" : "would", "don" : "do", "doesn" : "does", "didn" : "did", "couldn" : "could", "shouldn" : "should", "mightn" : "might", "mustn" : "must", "ain" : "I am",  } ## Need a huge dictionary

STOPSET = {"a", "as", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again", "against", "aint", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "arent", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "cmon", "cs", "came", "can", "cant", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", 
           "could", "couldnt", "course", "currently", "definitely", "described", "despite", "did", "didnt", "different", "do", "does", "doesnt", "doing", "dont", "done", "down", "downwards", "during", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "far", "few", "ff", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "had", "hadnt", "happens", "hardly", "has", "hasnt", "have", "havent", "having", "he", "hes", "hello", "help", "hence", "her", "here", "heres", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "id", "ill", "im", "ive", 
           "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isnt", "it", "itd", "itll", "its", "its", "itself", "just", "keep", "keeps", "kept", "know", "knows", "known", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "little", "look", "looking", "looks", "lot", "ltd", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "mr", "mrs", "ms", "much", "must", "my", "myself", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", 
           "ourselves", "out", "outside", "over", "overall", "own", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides", "que", "quite", "qv", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldnt", "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "ts", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "thats", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "theres", "thereafter", 
           "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "theyd", "theyll", "theyre", "theyve", "thing", "things", 
           "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "value", "various", "very", "via", "viz", "vs", "want", "wants", "was", "wasnt", "way", "we", "wed", "well", "were", "weve", "welcome", "well", "went", "were", "werent", "what", "whats", "whatever", "when", "whence", "whenever", "where", "wheres", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whos", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without", "wont", "wonder", "would", "would", "wouldnt", "yes", "yet", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves", "zero"};
#'''
#STOPSET = set(stopwords.words('english'))

STOPSET_SPANISH = {'de', 'que', 'del', 'al', 'se', 'este', 'pa', 'un', 'una', 'unas', 'unos', 'uno', 'sobre', 'todo', 'tambien', 'tras', 'otro', 'algun', 'alguno', 'alguna', 'algunos', 'algunas', 'ser', 'es', 'soy', 'eres', 'somos', 'sois', 'estoy', 'esta', 'estamos', 'estais', 'estan', 'como', 'en', 'para', 'atras', 'porque', 'por que', 'estado', 'estaba', 'ante', 'antes', 'siendo', 'ambos', 'pero', 'por', 'poder', 'puede', 'puedo', 'podemos', 'podeis', 'pueden', 'fui', 'fue', 'fuimos', 'fueron', 'hacer', 'hago', 'hace', 'hacemos', 'haceis', 'hacen', 'cada', 'fin', 'incluso', 'desde', 'primero', 'conseguir', 'consigo', 'consigue', 'consigues', 'conseguimos', 'consiguen', 'ir', 'voy', 'va', 'vamos', 'vais', 'van', 'vaya', 'gueno', 'ha', 'tener', 'tengo', 'tiene', 'tenemos', 'teneis', 'tienen', 'el', 'la', 'lo', 'las', 'los', 'su', 'aqui', 'mio', 'tuyo', 'ellos', 'ellas', 'nos', 'nosotros', 'vosotros', 'vosotras', 'si', 'dentro', 'solo', 'solamente', 'saber', 'sabes', 'sabe', 'sabemos', 'sabeis', 'saben', 'ultimo', 'largo', 'bastante', 'haces', 'muchos',
                   'aquellos', 'aquellas', 'sus', 'entonces', 'tiempo', 'verdad', 'verdadero', 'verdadera', 'cierto', 'ciertos', 'cierta', 'ciertas', 'intentar', 'intento', 'intenta', 'intentas', 'intentamos', 'intentais', 'intentan', 'dos', 'bajo', 'arriba', 'encima', 'usar', 'uso', 'usas', 'usa', 'usamos', 'usais', 'usan', 'emplear', 'empleo', 'empleas', 'emplean', 'ampleamos', 'empleais', 'valor', 'muy', 'era', 'eras', 'eramos', 'eran', 'modo', 'bien', 'cual', 'cuando', 'donde', 'mientras', 'quien', 'con', 'entre', 'sin', 'trabajo', 'trabajar', 'trabajas', 'trabaja', 'trabajamos', 'trabajais', 'trabajan', 'podria', 'podrias', 'podriamos', 'podrian', 'podriais', 'yo', 'aquel' }


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def lemmatize_tokens(tokens, lemmatizer):
    stemmed = []
    for item in tokens:
        stemmed.append(lemmatizer.lemmatize(item))
    return stemmed

  
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in STOPSET_SPANISH and len(i) > 1]
    #tokens = stem_tokens(tokens, PorterStemmer())
    tokens = lemmatize_tokens(tokens, WordNetLemmatizer())
    return tokens  


def clean_text(text):
    text = parser.unescape(text)
    text = text.encode('ascii', errors='ignore').decode('utf8')
    text = regex.sub('\\n', ' ', text)  # @UndefinedVariable
    text = regex.sub('\\b(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]', '', text)#, flags=regex.MULTILINE)  # @UndefinedVariable
    text = regex.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)','', text)  # @UndefinedVariable
    text = regex.sub("'"," '",text)  # @UndefinedVariable
    words = text.split()
    transformed_words = [APOSTROPHES[word] if word in APOSTROPHES else word for word in words]
    text = " ".join(transformed_words)
    text = " ".join(regex.findall('[A-Z][^A-Z]*', text.title()))  # @UndefinedVariable
    text = "".join(''.join(s)[:2] for _, s in itertools.groupby(text))
    text = "".join([ch for ch in text if ch not in punctuation])
    text = regex.sub(' +',' ',text)  # @UndefinedVariable
    return text.lower()


def read_data(path, reviews, labels, words):
    for root, _, files in os.walk(path, topdown=True):  
        for name in files:
            f = open(os.path.join(root, name), 'r', encoding="utf8")
            content = f.read()
            clean_content = clean_text(content)
            [words.append(w) for w in clean_content.split()]     
            reviews.append(clean_content)
            labels.append(path==path_pos)            
   

def create_wordcloud(text, max_nr_words, wordcloud_name, stopwords = STOPSET):
    wordcloud = WordCloud(stopwords = stopwords,
                          max_words=max_nr_words,
                          width=3000,
                          height=2000
                          ).generate(text)
    wordcloud.to_file(os.path.join(os.path.dirname(__file__), wordcloud_name))
    #plt.imshow(wordcloud)
    #plt.axis("off")
    #plt.ion()
    #plt.show()


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def wordcloud_per_topic(model, feature_names, n_top_words, model_name):
  for topic_idx, topic in enumerate(model.components_):
    topic_dist_idx = topic.argsort()[:-n_top_words-1:-1]
    topic_dist = topic[topic_dist_idx]
    norm_fac = 100/sum(topic_dist)
    words_topic = []
    names = [feature_names[i] for i in topic_dist_idx]
    for ele in range(len(topic_dist)):
            words_topic += [names[ele]]*int(topic_dist[ele]*norm_fac)
    text = " ".join(words_topic)
    save_path = "./" + model_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)           
    wordcloud_name = save_path + "/Wordcloud_topic_" + str(topic_idx) +".png"
    create_wordcloud(text, n_top_words, wordcloud_name)


n_features = 2000
n_topics = 5
n_top_words = 30

labels = []
reviews = []
words =[]
path_docs = os.path.dirname(os.path.abspath(__file__)) + '\\Documents'


print("\nLoading and cleaning the text in the reviews...")
t0 = time()
#read_data(path_neg, reviews, labels, words)
#read_data(path_pos, reviews, labels, words)
read_data(path_docs, reviews, labels, words)
data_samples = reviews
n_samples = len(data_samples)
print("done in %0.3fs." % (time() - t0))

'''#Test
path_test = os.path.dirname(os.path.abspath(__file__)) + '\\Test'
read_data(path_test, reviews, labels, words)
text = " ".join(reviews)
text = " ".join(tokenize(text))
wordcloud_name = "./Wordcloud_Test.png"
create_wordcloud(text, n_top_words, wordcloud_name, stopwords = STOPSET_SPANISH)
#'''

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.75, min_df=0.05, #max_df=0.95, min_df=0.01,
                                   max_features=n_features,
                                   tokenizer=tokenize, stop_words=STOPSET)
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.75, min_df=0.05, #max_df=0.95, min_df=0.01,
                                max_features=n_features,
                                tokenizer=tokenize, stop_words=STOPSET)
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print()


# Fit the NMF model
print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)
wordcloud_per_topic(nmf, tfidf_feature_names, n_top_words,'NMF')


# Fit the LDA model
print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
#lda.fit(tf)
lda.fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
#print_top_words(lda, tf_feature_names, n_top_words)
print_top_words(lda, tfidf_feature_names, n_top_words)
wordcloud_per_topic(lda, tfidf_feature_names, n_top_words, 'LDA')
#'''
    