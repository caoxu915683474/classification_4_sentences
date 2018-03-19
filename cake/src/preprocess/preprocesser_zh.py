import sys
import re
import html
import codecs
import string
import math

from hanziconv import HanziConv
from tqdm import tqdm, trange
from stanfordcorenlp import StanfordCoreNLP

import cake.src.preprocess.preprocesser as preprocesser

def replace_numbers(text):
    """ Replace number to number_token
    
    - Doctest:
        >>> replace_numbers("123456.")
        '#number'
        >>> replace_numbers("12.232")
        '#number'
        >>> replace_numbers(".232")
        '#number'
        >>> replace_numbers("he232llo!")
        'he#numberllo!'
    """
    # return re.sub(r"\d+[\.|(\.\d+)]?", "#number", text)
    return re.sub(r"[-+]?\d*\.\d+|\d+\.?", "#number", text)

def remove_html_tags(text):
    """ Remove html tags

    - Doctest:
        >>> remove_html_tags("<tag><imgsrc=?> https://google.com")
        ' .  https://google.com'
        >>> remove_html_tags("[color=#123123] Hello!")
        ' .  Hello!'
    """
    text = re.sub(r"\[/?[a-z]+?(=.*?)?\]", " . ", text) # Color tags/ b tags/ emoji
    return re.sub(r"( ?\.+ )+", " . ", re.sub(r"<[^>]*>", " . ", text)) # xml tags

def process_urls_link(text, sub):
    """ Remove url web links

    -Doctest:
        >>> process_urls_link("详见链接：https://google.com.tw", "")
        '详见链接：'
        >>> process_urls_link("请至ai.chuangxin.com", "")
        '请至'
        >>> process_urls_link("公司IP为 10.18.125.12", "")
        '公司IP为 '
    """
    return re.sub(r"(http(s)?(:\/\/))?(www\.)?[a-zA-Z0-9-_\.]+(\.[a-zA-Z0-9]{2,})([-a-zA-Z0-9:%_\+.~#?&//=]*)", sub, text)

def to_half_width(text):
    """
    Transform the full width to half width.

    -Doctest:
        >>> to_half_width("今天，天气、如何。？")
        '今天，天气、如何。?'
    """    
    processed = ""
    for word in text:
        if not (word==u"，" or word==u"、" or word==u"。"):
            inside_code = ord(word)
            if inside_code == 12288:
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):
                inside_code -= 65248
            word = chr(inside_code)
        processed += word
    return processed

def remove_punctuations(text):
    """
    Remove punctuations except {，。、!;?}
    """
    return "".join(re.findall(u"[\u4e00-\u9fa5|0-9|a-z|A-Z|，|。|、|!|;|?]+", text))

def remove_custom_patterns(text, custom_patterns):
    """
    Remove custom defined regex patterns Ex: (r".*?", ""), (r"apple+?", "apl")
    """
    for pattern in custom_patterns:
        if "\\\\" in pattern[0]:
            p = repr(pattern[0]).replace("\'","")
        else:
            p = pattern[0]
        text = re.sub(p, pattern[1], text)
    return text

def zh_ts_convert(text, t2s=True):
    if t2s: # traditional to simplified
        return HanziConv.toSimplified(text)
    else:
        return HanziConv.toTraditional(text)

class ZhPreprocesser(preprocesser.Preprocesser):
    def __init__(
        self,
        lowercase=True,
        remove_html=True,
        remove_or_replace_urls="",
        half_width=True,
        remove_punctuation=True,
        custom_pattern_path=[],
        stanford_corenlp_path="",
        char_segmentation=False,
        replace_ner=True,
        replace_number=True,
        convert_t2s=True,
        convert_s2t=False
    ):

        self.remove_html = remove_html
        self.remove_or_replace_urls = remove_or_replace_urls
        self.remove_punctuation = remove_punctuation
        self.custom_pattern_path = custom_pattern_path
        self.replace_ner = replace_ner
        self.stanford_corenlp_path = stanford_corenlp_path
        self.char_segmentaion = char_segmentation
        self.half_width = half_width
        self.replace_number = replace_number
        self.lowercase = lowercase
        self.convert_t2s = convert_t2s
        self.convert_s2t = convert_s2t

        self.stanford_corenlp = StanfordCoreNLP(self.stanford_corenlp_path, lang="zh", memory="8g")

    def tag_corpus_ner(self, corpus):
        """
        Tag named entitties in corpus with stanfordNER toolkit
        """
        print("\ntagging sentences with Stanford NER...")
        tagged_corpus = []
        for i in tqdm(range(len(corpus))):
            # print("\r%s " % i, end="")
            if not sent:
                tagged_corpus.append([])
            else:
                tagged_corpus += [self.stanford_corenlp.ner(corpus[i])]

        # get dictionary of named entities per document
        named_entities = []
        for tagged_doc in tagged_corpus:
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

        return tagged_corpus, named_entities

    def replace_ner_entities(self, tagged_corpus):
        corpus = []
        for d in range(len(tagged_corpus)):
            line = []
            for entity in tagged_corpus[d]:
                if entity[1] == "O" or entity[1] == "MISC":
                    line.append(entity[0])
                else:
                    line.append("#"+entity[1])

            corpus.append(" ".join(line))
        return corpus

    def segmentation(self, corpus, char_level=False):
        """
        Chinese word segmentation w/t StanfordCoreNLPServer
        """
        print("\nsegmentating sentences in corpus")
        segged_sentences = []
        for i in tqdm(range(len(corpus))):
            if not char_level:
                segged_sent = self.stanford_corenlp.word_tokenize(corpus[i]) if corpus[i]!="" else [""]
            else:
                segged_sent = list(corpus[i])
            segged_sentences.append(" ".join(segged_sent))
        return segged_sentences

    def preprocess_corpus(self, corpus):
        """
        Preprocess main function.
        """
        self.segment_batch = int(math.ceil(len(corpus)/25000))
        self.ner_batch = int(math.ceil(len(corpus)/5000))
        print("preprocessing corpus...")
        print("corpus size:", len(corpus))

        # first pass over the corpus: prepare for NER
        print("first pass over the corpus...\n\tunescape characters")
        if self.remove_html: print("\tremove html")
        print("\treplacing URLs to %s" % self.remove_or_replace_urls)
        if self.half_width: print("\tto half width encoding")
        if self.remove_punctuation: print("\tremove_punctuations")
        if self.convert_s2t: print("\tconvert simplified to tradit'nal")
        if self.convert_t2s: print("\tconvert tradit'nal to simplified")
        if len(self.custom_pattern_path) > 0: print("\tsubstitute custom_patterns")
        for d in tqdm(range(len(corpus))):
            corpus[d] = html.unescape(corpus[d])+" "
            if self.remove_html:
                corpus[d] = remove_html_tags(corpus[d])
            corpus[d] = process_urls_link(corpus[d], self.remove_or_replace_urls)
            if len(self.custom_pattern_path) > 0:
                corpus[d] = remove_custom_patterns(corpus[d], self.custom_pattern_path)
            if self.half_width:
                corpus[d] = to_half_width(corpus[d])
            if self.remove_punctuation:
                corpus[d] = remove_punctuations(corpus[d])
            if self.convert_t2s:
                corpus[d] = zh_ts_convert(corpus[d])
            if self.convert_s2t:
                corpus[d] = zh_ts_convert(corpus[d], False)

        if self.replace_ner:
            _, tagged_corpus, named_entities = self.tag_corpus_ner(corpus)
            print("\n\nReplace named entities as #[EntityClass]")   
            corpus = self.replace_ner_entities(tagged_corpus)

            # TODO: Do we need this feature as an option??
                # print("merging named entities as single tokens...")

            # debug NER
            fw = codecs.open("debug_NER.txt", "w", "utf-8")
            for tags in named_entities:
                fw.write("%s\n" % list(tags.items()))
            fw.close()
        else:
            corpus = self.segmentation(corpus, self.char_segmentaion)

        # second pass over the corpus: ready to export
        print("\nsecond pass over the corpus...")
        if self.replace_number: print("\treplace number")
        if self.lowercase: print("\tconvert to lowercase")
        for d in tqdm(range(len(corpus))):
            if self.replace_number:
                corpus[d] = replace_numbers(corpus[d])
            if self.lowercase:
                corpus[d] = corpus[d].lower()

        return corpus

# Unit Test
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    def ZhPreprocesserTest():
        
        import csv
        import pandas as pd
        from src.preprocess.preprocesser_en import EnPreprocesser
        from src.preprocess.preprocesser_zh import ZhPreprocesser

        print("Running ZhPreprocesser Unit Test")
        train = pd.read_csv("./data/train", converters={"comment":str})
        tp = ZhPreprocesser(
            lowercase=True,
            remove_html=True,
            remove_or_replace_urls="",
            half_width=True,
            custom_pattern_path=[(r"吃屎长大.*? ", ""), (r"\n", "")],
            replace_ner=False,
            stanford_corenlp_path="/home/cxpc/Documents/toolkits/stanfordNLP/stanford-corenlp-full-2017-06-09/",
            replace_number=True
        )
        parsed_corpus = tp.preprocess_corpus(train["comment"].tolist()[:500])
        train["comment"] = parsed_corpus
        train.to_csv("./data/train.csv", quoting=csv.QUOTE_NONNUMERIC, index=False)
        #TODO: need any assert?

    ZhPreprocesserTest()
