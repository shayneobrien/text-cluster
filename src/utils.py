import re

def clean_string(string):
    """ Clean string and rejoin it separating for stopwords appropriately """
    tokens = flatten([clean_token(t) for t in string.split()])
    return ' '.join([t for t in tokens if t not in stopwords])

def clean_token(token):
    """ Remove everything but whitespace, the alphabet, digits;
    separate apostrophes for stopwords; replace only digit tokens """
    if is_digits(token):
        token = '<NUM>'
    elif is_unk(token):
        token = '<UNK>'
    elif is_noise(token):
        token = '<NOS>'
    else:
        token = re.sub(r"['-]+", ' ', token)
        token = re.sub(r"[^a-z0-9-\s]", '', token.lower())
    return token.split()

def get_stopwords(filename='../data/stopwords.txt'):
    return read_file(filename)

def make_stopwords(filler_file='../data/filler-words.txt',
                    terms_file='../data/top-terms.txt'):
    return read_file(filler_file) + read_file(terms_file)

def read_file(filename):
    with open(filename, 'r') as f:
        entries = f.read().split('\n')
    return entries

def is_digits(token):
    """ Check if a string has any numbers in it """
    return bool(re.search(r'\d', token))

def is_unk(token):
    """ Check if the token is an unk token """
    return bool(r'<unk>' in token)

def is_noise(token):
    """ Check if the token indicates noise """
    return bool(r'[noise]' in token)

def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]

stopwords = get_stopwords()
