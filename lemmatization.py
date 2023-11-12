
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer




def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
lemmatizer = WordNetLemmatizer()
def lemmatize_tweet(tweet):
    nltk_tagged = nltk.pos_tag(word_tokenize(tweet))
    lemmatized_words = [lemmatizer.lemmatize(word, nltk_tag_to_wordnet_tag(tag))
                        if nltk_tag_to_wordnet_tag(tag) is not None else word for word, tag in nltk_tagged]
    return ' '.join(lemmatized_words)
