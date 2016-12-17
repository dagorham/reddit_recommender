from nltk.stem.porter import PorterStemmer

__name__ = 'text_processor'


def process_text(string_input, punc, english_stop_words, stem=True):
    """
    :param string_input: string to be processed
    :param punc: punctuation to be removed
    :param english_stop_words: stop words to be removed
    :param stem: tells whether to stem words or not
    :return: processed string, with everything lowercase, no punctuation, and stemmed if requested
    """

    if len(string_input) > 10000:
        string_input = ' '.join(string_input.split()[:10000])

    for character in string_input:
        if character in punc:
            string_input.replace(character, "")

    processed_string = string_input.lower().split()

    if stem:
        stemmer = PorterStemmer()  # instantiate stemmer
        processed_string = ' '.join([stemmer.stem(word) for word in processed_string if word not in english_stop_words])
        return processed_string

    processed_string = ' '.join([word for word in processed_string if word not in english_stop_words])

    return processed_string
