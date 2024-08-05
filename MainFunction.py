#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import sys
import nltk
from pprint import pprint

# Define class for tokenization
class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        Input format: a paragraph of text
        Output format: a list of lists of words.
        e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences

# Define function for sentiment scoring
def value_of(sentiment):
    if sentiment == 'positive':
        return 1
    elif sentiment == 'negative':
        return -1
    elif sentiment == 'inc':
        return 5
    return 0

def sentence_score(sentence_tokens, previous_token, acum_score):
    if not sentence_tokens:
        return acum_score
    else:
        current_token = sentence_tokens[0]
        tags = current_token[2]
        token_score = sum([value_of(tag) for tag in tags])
        if previous_token is not None:
            previous_tags = previous_token[2]
            if 'inc' in previous_tags:
                token_score *= 2.0
            elif 'dec' in previous_tags:
                token_score /= 2.0
            elif 'inv' in previous_tags:
                token_score *= -1.0
        return sentence_score(sentence_tokens[1:], current_token, acum_score + token_score)

def sentiment_score(review):
    return sum([sentence_score(sentence, None, 0.0) for sentence in review])

# Define function for handling usage information
def usage():
    """ This function prints the usage information """
    print("Usage: python script.py <file>")
    sys.exit(1)

# Main function to process input and analyze sentiment
def main():
    if len(sys.argv) == 1:
        usage()
    else:
        if sys.argv[1] == "-":
            text = sys.stdin.read()
        else:
            try:
                file = sys.argv[1]
                with open(file, 'r') as fd:
                    text = fd.read()
            except:
                print("ERROR: Input file could not be opened.")
                usage()

        splitter = Splitter()
        postagger = POSTagger()
        dicttagger = DictionaryTagger([
            'dicts/positive.yml', 'dicts/negative.yml',
            'dicts/inc.yml', 'dicts/dec.yml', 'dicts/inv.yml'
        ])

        splitted_sentences = splitter.split(text)
        pprint(splitted_sentences)

        pos_tagged_sentences = postagger.pos_tag(splitted_sentences)
        pprint(pos_tagged_sentences)

        dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
        pprint(dict_tagged_sentences)

        print("Analyzing sentiment...")
        score = sentiment_score(dict_tagged_sentences)
        if score < 0:
            print("Document contents Negativity")
        elif score == 0:
            print("Document neutral")
        elif score > 0:
            print("Document contents positivity")
        print(score)

if __name__ == "__main__":
    main()

