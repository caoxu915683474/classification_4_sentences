preprocess = {
    "zh": {
        "lang": "zh",
        "lowercase": True,
        "remove_html": True,
        "remove_or_replace_urls": "",
        "half_width": True,
        "custom_pattern_path": [],
        "replace_ner": False,
        "stanford_corenlp_path": "",
        "replace_number": True
    },
    "en": {
        "lang": "en",
        "lowercase": True,
        "remove_html": True,
        "join_urls": True,
        "use_bigrams": False,
        "use_ner": True,
        "stanford_ner_path": "",
        "use_lemmatizer": True,
        "use_stemmer": False,
    }
}

def printer(cfg):
    print("")
    print("\nConfiguration: Preprocess")
    for c in cfg.items():
        print("    %s : %s" % (c[0], c[1]))
    print("")
