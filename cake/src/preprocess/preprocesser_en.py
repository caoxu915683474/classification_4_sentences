# coding:utf-8
import re, os
import unicodedata
import codecs
import html
import nltk
import csv
import string
import math
from tqdm import tqdm, trange
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.collocations import *
from nltk.tag import StanfordNERTagger
from sklearn.feature_extraction.text import CountVectorizer

import cake.src.preprocess.preprocesser as preprocesser
import warnings

class CustomTokenizer(object):
    def __init__(self, tokenizer, stemmer, token_pattern, numeric_pattern):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.token_pattern = token_pattern
        self.numeric_pattern = numeric_pattern
    def __call__(self, doc):
        tokens = []
        for t in self.tokenizer(doc):
            #print(t)
            #input()
            if self.token_pattern.match(t) and not self.numeric_pattern.match(t):
                while "_" in t:
                    splt = t.split("_")
                    t = "".join(splt[1:])
                    tokens.append(self.stemmer(splt[0]))
                tokens.append(self.stemmer(t))
        return tokens

class EnPreprocesser(preprocesser.Preprocesser):
    def __init__(self, 
        strip_accents="unicode",
        lowercase=True,
        remove_html=True,
        join_urls=True,
        use_bigrams=True,
        use_ner=True,
        stanford_ner_path="",
        use_lemmatizer=False,
        use_stemmer=False
        ):

        self.stanford_ner_path = stanford_ner_path      # path to stanford NER
        self.strip_accents = strip_accents              # options: {‘ascii’, ‘unicode’, None}
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.join_urls = join_urls  
        self.use_bigrams = use_bigrams
        self.use_ner = use_ner
        self.use_lemmatizer = use_lemmatizer            # use lemmatizer instead of stemmer?
        self.use_stemmer = use_stemmer

        # self.stanford_corenlp = StanfordCoreNLP(self.stanford_corenlp_path, memory="8g")
        self.sentence_splitter = PunktSentenceTokenizer().tokenize      # Punkt sentence splitter
        self.stemmer = SnowballStemmer("english").stem                  # Snowball stemmer
        self.lemmatizer = WordNetLemmatizer().lemmatize                 # WordNet lemmatizer
        self.base_tokenizer = CountVectorizer().build_tokenizer()       # sklearn tokenizer works the best, I think...
        self.stop_words = stopwords.words("english")                    # nltk list of 128 stopwords
        self.token_pattern = re.compile(r"(?u)\b(\w*[a-zA-Z_]\w+|\w+[a-zA-Z_]\w*)\b")   # default value was r"(?u)\b\w\w+\b"
        self.numeric_pattern = re.compile(r"^[0-9]+$")                  # number regex
        self.url_pattern = re.compile(r"((http://)?(www\..*?\.\w+).*?)\s")
        self.compound_pattern = re.compile(r"\w+(\-\w+)+")

        if self.use_lemmatizer:
            self.tokenizer = CustomTokenizer(self.base_tokenizer, self.lemmatizer, self.token_pattern, self.numeric_pattern)
        elif self.use_stemmer:
            self.tokenizer = CustomTokenizer(self.base_tokenizer, self.stemmer, self.token_pattern, self.numeric_pattern)
        else:
            self.tokenizer = CustomTokenizer(self.base_tokenizer, lambda x: x, self.token_pattern, self.numeric_pattern)
        

    def find_nbest_bigrams(self, corpus, n, metric, min_freq):
        """
        Find the top-N most frequently occurring bigrams within the corpus.
        """
        print("\nfinding top-%d bigrams using %s..." % (n, metric))
        alltokens = []
        simplerTokenizer = CustomTokenizer(self.base_tokenizer, lambda x: x, re.compile(".*"), re.compile("^$"))
        for doc in corpus:
            for token in [t for t in simplerTokenizer(doc)]:
                alltokens.append(token)
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(alltokens)
        finder.apply_freq_filter(min_freq) # bigrams must appear at least 5 times
        if metric.lower() == "pmi":
            best_bigrams = finder.nbest(bigram_measures.pmi, n)  # doctest: +NORMALIZE_WHITESPACE
        elif metric.lower() == "chi_sq":
            best_bigrams = finder.nbest(bigram_measures.chi_sq, n)  # doctest: +NORMALIZE_WHITESPACE
        else:
            raise Exception("Unknown metric for bigram finder")
        return best_bigrams

    def remove_punctuation(self, text):
        """
        Remove punctuation.
        """
        return "".join(re.findall(r"[a-zA-Z0-9\s]+", text))
        # return "".join(re.findall(r"[a-zA-Z0-9,.;!:'?\s]+", tokens))
        # return tokens

    def tag_corpus_ner(self, corpus):
        """
        Tag named entitties in corpus with stanfordNER toolkit
        """

        if not hasattr(self, "stanford_ner"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                # import imp
                self.stanford_ner = StanfordNERTagger(
                    self.stanford_ner_path+"classifiers/english.conll.4class.distsim.crf.ser.gz",
                    self.stanford_ner_path+"stanford-ner.jar"
                )
                self.stanford_ner._stanford_jar = self.stanford_ner_path+"stanford-ner.jar:"+self.stanford_ner_path+"lib/*"
        
        print("splitting sentences in corpus (for NER)...")
        corpus_sentences = []
        sentence_to_doc_map = {}
        sent_no = 0
        for d in tqdm(range(len(corpus))):
            # print("\r%s " % d, end="")
            for sent in self.sentence_splitter(corpus[d]):
                corpus_sentences.append(sent)
                sentence_to_doc_map[sent_no] = d
                sent_no += 1
        tokenized_sentences = []
        for sent in corpus_sentences:
            tokenized_sentences.append([t for t in re.split(r"\s+", sent) if len(t) > 0])
            #tokenized_sentences = [re.split(r'\s+', sent) for sent in corpus_sentences]
        
        print("tagging sentences with Stanford NER...")
        tagged_sentences = []
        for batch in tqdm(range(self.ner_batch)):
            # print("\r%s/%s tagging sentences with Stanford NER..." % (batch, self.ner_batch), end="")
            chunk = int(len(corpus)/self.ner_batch)
            tagged_sentences += self.stanford_ner.tag_sents(tokenized_sentences[batch*chunk:(batch+1)*chunk])
        # process NER output
        tagged_corpus = []
        current_doc_no = 0
        current_doc = []
        for i in range(len(tagged_sentences)):
            doc_no = sentence_to_doc_map[i]
            if doc_no == current_doc_no:
                current_doc += tagged_sentences[i]
            else:
                tagged_corpus.append(current_doc)
                current_doc = []
                current_doc_no = doc_no
        tagged_corpus.append(current_doc)

        # get dictionary of named entities per document
        named_entities = []
        for tagged_doc in tagged_sentences:
            tags = {}
            current_ne = []
            for token, tag in tagged_doc:
                if current_ne:
                    if tag == "O" or (tag != "O" and tag != current_ne[-1][1]):
                        tags[" ".join([t for t,_ in current_ne])] = current_ne[0][1]
                        current_ne = []
                if tag != "O":
                    current_ne.append((token, tag))
            if current_ne:
                tags[" ".join([t for t,_ in current_ne])] = current_ne[0][1]
            named_entities.append(tags)
        return tagged_sentences, named_entities


    def preprocess_corpus(self, corpus):
        """
        Preprocess the corpus.
        """
        self.ner_batch = int(math.ceil(len(corpus)/5000))
        print("preprocessing corpus...")
        print("corpus size: %i, ner_batch=%i" % (len(corpus), self.ner_batch))

        # first pass over the corpus: prepare for NER
        print("first pass over the corpus...\n\tunescape characters")
        if self.remove_html: print("\tremove html")
        if self.strip_accents: print("\tstrip accents")
        if self.join_urls: print("\tjoin URLs")
        print("\tjoin compound words\n\tspace out punctuation")
        
        for d in tqdm(range(len(corpus))):
            corpus[d] = html.unescape(corpus[d])+" "
            
            if self.remove_html:
                corpus[d] = self.remove_html_tags(corpus[d])
            if self.strip_accents == "unicode":
                corpus[d] = self.strip_accents_unicode(corpus[d])
            if self.join_urls:
                corpus[d] = self.join_urls_to_token(corpus[d], self.url_pattern)
            corpus[d] = self.join_compound_words(corpus[d], self.compound_pattern)
            corpus[d] = self.space_out_punctuation(corpus[d])
            # print("\r\t%s" % d,  end="")

        if self.use_ner:
            tagged_corpus, named_entities = self.tag_corpus_ner(corpus)

            # debug NER
            fw = codecs.open("debug_NER.txt", "w", "utf-8")
            for tags in named_entities:
                fw.write("%s\n" % list(tags.items()))
            fw.close()

            print("\nmerging named entities as single tokens...")
            for d in tqdm(range(len(tagged_corpus))):
                tags = named_entities[d]
                for ne in tags:
                    corpus[d] = corpus[d].replace(ne, re.sub(r"\s+", "", ne))
                # print("\r%s " % d,  end="")
        
        # second pass over the corpus: remove punctuation and convert to lowercase 
        # (these were useful above for NER, but now can be removed)
        print("\nsecond pass over the corpus...")
        if self.lowercase: print("\tconvert to lowercase")
        print("\tremove punctuation")
        for d in tqdm(range(len(corpus))):
            corpus[d] = self.remove_punctuation(corpus[d])
            if self.lowercase:
                corpus[d] = corpus[d].lower()
            # print("\r\t%s" % d,  end="")
        
        if self.use_bigrams:
            # find top N bigrams
            # best_bigrams = self.find_nbest_bigrams(corpus, 100, "pmi", 10)
            best_bigrams = self.find_nbest_bigrams(corpus, 100, "chi_sq", 10)
            
            # debug bigrams
            fw = codecs.open("debug_bigrams.txt", "w", "utf-8")
            for w1, w2 in best_bigrams:
                fw.write(w1+" "+w2+"\n")
            fw.close()

            print("\n")
            for d in range(len(corpus)):
                print("\r%s merging bigrams as single tokens..." % d,  end="")
                for w1, w2 in best_bigrams:
                    corpus[d] = corpus[d].replace(w1+" "+w2, w1+w2)

        return [sent for sent in corpus]

    # helper functions
    def strip_accents_unicode(self, text):
        return "".join([c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)])

    def remove_html_tags(self, text):
        return re.sub(r"( ?\.+ )+", " . ", re.sub(r"<[^>]*>", " . ", text))

    def join_urls_to_token(self, text, url_pattern):
        m = re.search(url_pattern, text)
        while m:
            text = re.sub(url_pattern, m.group(3).replace("http://","").replace(".",""), text)
            m = re.search(url_pattern, text)
        return text

    def join_compound_words(self, text, compound_pattern):
        m = re.search(compound_pattern, text)
        while m:
            text = re.sub(m.group(0), m.group(0).replace("-",""), text)
            m = re.search(compound_pattern, text)
        return text

    def space_out_punctuation(self, text):
        text = re.sub(r",\s", " , ", text)
        text = re.sub(r"\.\.\.\s", " ... ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r";\s", " ; ", text)
        text = re.sub(r":\s", " : ", text)
        text = re.sub(r"\?\s", " ? ", text)
        text = re.sub(r"!\s", " ! ", text)
        text = re.sub(r"\"", " \" ", text)
        text = re.sub(r"\'", " \' ", text)
        text = re.sub(r"\s\(", " ( ", text)
        text = re.sub(r"\)\s", " ) ", text)
        text = re.sub(r"\s\[", " [ ", text)
        text = re.sub(r"\]\s", " ] ", text)
        text = re.sub(r"-", " - ", text)
        text = re.sub(r"_", " _ ", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\r", " ", text)
        text = re.sub(r"\s+", " ", text)
        tokens = self.tokenizer(text)
        tokens = " ".join(tokens)
        return tokens

# Unit Test
if __name__ == "__main__":

    import pandas as pd
    import csv
    from src.preprocess.preprocesser_en import EnPreprocesser
    from src.preprocess.preprocesser_zh import ZhPreprocesser
    
    print("Running EnPreprocesser Unit Test")
    train = pd.read_csv("./data2/train", converters={"comment":str})
    tp = EnPreprocesser(
        lowercase=True,
        remove_html=True,
        join_urls=True,
        use_bigrams=False,
        use_ner=True,
        stanford_ner_path="/home/cxpc/Documents/toolkits/stanfordNLP/stanford-ner/",
        use_lemmatizer=True,
        use_stemmer=False,
    )
    parsed_corpus = tp.preprocess_corpus(train["comment"].tolist()[:5000])
    train["comment"][:5000] = parsed_corpus
    train.to_csv("./data/train.csv", quoting=csv.QUOTE_NONNUMERIC, index=False)
