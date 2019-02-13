import re
import csv
import nltk
from math import log
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from collections import OrderedDict
from nltk.util import ngrams

input_file = "dataset_tema3.csv"

#Minimum number of characters for a text to compute its summary
min_size = 10

#Columns names inside the csv file
headline_tag = "headlines"
text_tag = "text"
ctext_tag = "ctext"
    
#List of punctuation
punct = list(punctuation)

#Punctuation for the end of sentence
end_sentence = ['.', '!', '?']
    
#Constant for title similarity
title_importance = 0.8

#Constant for sentences position
most_important = 0.5
second_important = 0.2

#Original sentences of the textes stored for computing the summary
original_corrected_sentences = []

'''
    Read the data from the input file and save them sorted on columns.
    In the end, components will store an array for each column inside the
    csv file: "headlines", "ctext" and "text".
'''
def read_dataset(components):
    with open(input_file) as f:
        reader = csv.DictReader(f)

        for row in reader:
            for (col, value) in row.items():
                if col not in components:
                    components[col] = []

                components[col].append(value)

'''
    Add extra spaces between punctuation and words to make sure that words will
    be correctly tokenized and punctuation is not merged to words.
'''
def add_spaces(text):
    text_len = len(text)

    new_text = ''

    for i in range(0, text_len - 1):
        new_text += text[i]

        #If punctuation is merged to words, add an extra space
        if text[i].isalpha() and text[i + 1] in punct:
            new_text += ' '

        #If punctuation is merged in the beginning of word, add extra space
        if text[i] in punct and text[i + 1].isalpha():
            new_text += ' '

    #Add the last character inside the text
    new_text += text[text_len - 1]

    return new_text

'''
    Remove wrong punctuation added in the original text.
'''
def remove_wrong_punctuation(text):
    #Split text in words    
    tokens = word_tokenize(text)

    tokens = [token.strip() for token in tokens]
    good_tokens = []

    nr_tokens = len(tokens)
    for i in range(len(tokens)):
        if i < (nr_tokens - 1):

            #Check if end of sentence was marked without starting new sentence
            if tokens[i] in end_sentence and (tokens[i + 1].isalpha() and 
                    not tokens[i + 1].istitle()):
                continue

        if i > 0:

            #Check if two consecutive punctuation were added
            if tokens[i] in punct and tokens[i - 1] in punct:
                continue

        good_tokens.append(tokens[i])

    return good_tokens

'''
    Save the original sentences inside the text, without the stopwords removed
    and without lemmatization, in order to use them when providing the final
    summary.
'''
def save_original_sentences(new_text):
    global original_corrected_sentences
   
    #Split text in sentences
    sentences = sent_tokenize(new_text)
    
    #Create new dictionary of sentences for current text
    dict_sentences = {}
    for i in range(len(sentences)):
        dict_sentences[i] = sentences[i]

    #Save the sentences for the current text
    original_corrected_sentences.append(dict_sentences)

'''
    Remove the stop words from the text.
'''
def remove_stopwords(tokens):
    stop_words = stopwords.words('english')

    good_tokens = []
    nr_tokens = len(tokens)

    #Keep only the tokens that are not stopwords
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            good_tokens.append(tokens[i].lower())

    return good_tokens

'''
    Lemmatize the words inside the text using nltk.
'''
def lemmatize_text(tokens):
    wnl = WordNetLemmatizer()
   
    #Itterate through the words and lemmatize all of them
    lemmatized_tokens = []
    for word in tokens:
        if word in punct:
            lemmatized_tokens.append(word)
        else:
            lemm_word = wnl.lemmatize(word)
            lemmatized_tokens.append(lemm_word)

    return lemmatized_tokens

'''
    Build the TF for the nouns inside the current text.
'''
def save_nouns(tokens, TFs):
    #Assign tags to all the words
    tags = nltk.pos_tag(tokens)

    tf = {}
    nr_words = 0

    for word,tag in tags:
        #Check if word is a noun
        if tag != None and tag.startswith('N'):
            nr_words +=1

            if word not in tf:
                tf[word] = 1
            else:
                tf[word] += 1 

    #print(nouns)
    #Normalize the TF
    for noun in tf:
        tf[noun] /= len(tokens)

    #Save the TF for the current text
    TFs.append(tf)

    return ' '.join(tokens)

'''
    Preprocess - clean the text before giving scores to words/sentences.
'''
def clean_text(text, TFs):

    #Add extra spaces before/after punctuation
    text = add_spaces(text)
    
    #Remove unnecesary information between paranthesis
    text = re.sub("[\(\[].*?[\)\]]", "", text)

    #Remove punctuation placed wrong in text
    tokens = remove_wrong_punctuation(text)

    #Save the original sentences
    text = ' '.join(tokens)
    save_original_sentences(text)
   
    #Remove stopwords and lemmatize the text
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)

    #Save the nouns and calculate TF
    text = save_nouns(tokens, TFs)
    
    return text

'''
    Create the TFIDF for each text in order to calculate sentences' scores.
'''
def build_tfidf(TFs, nr_textes):
    IDF = {}

    #Compute the IDF for the entire dataset
    for tf in TFs:
        #Check in how many textes a word appears
        for noun in tf:
            if noun in IDF:
                IDF[noun] += 1
            else:
                IDF[noun] = 1

    #Normalize IDF
    for idf in IDF:
        IDF[idf] = log(nr_textes/IDF[idf])

    #Calculate TFIDF for each text
    TFIDF = []
    for tf in TFs:

        tfidf = {}
        for noun in tf:
            tfidf[noun] = tf[noun] * IDF[noun]

        TFIDF.append(tfidf)

    return TFIDF

'''
    Create a dictionary to store the current score for each sentence.
'''
def create_sentences_dictionary(text):
    sentences = sent_tokenize(text)
    sentences_score = OrderedDict()

    #Iterate through the sentences and store 0 as current score
    for sent in sentences:
        words_sentence = word_tokenize(sent)
        sentences_score[tuple(words_sentence)] = 0.0

    return sentences_score

'''
    Calculate initial score of sentences according to the TFIDF of the words.
'''
def sentences_initial_score(sentences_score, tfidf):
    for sent in sentences_score:
        sentence_length = len(sent)

        sent_score = 0

        #Iterate through all the words of the sentence
        for word in sent:
            #Track the score for nouns
            if word in tfidf:
                sent_score += tfidf[word]

        #Normalize the score for the sentence
        sent_score /= sentence_length
        sentences_score[sent] += sent_score

    return sentences_score

'''
    Give additional score to sentences according to their similarity with the
    title.
'''
def title_similarity_score(sentences_score, title):
    title_words = [word.lower() for word in
            word_tokenize(title) if word not in punct]
    
    title_length = len(title_words)
    
    for sent in sentences_score:

        additional_score = 0
        
        #Iterate the words in the sentence
        for word in sent:
            #Count how many words from title appear in sentence
            if word in title_words:
                additional_score += 1

        #Normalize the score
        additional_score /= title_length
        additional_score *= title_importance

        sentences_score[sent] += additional_score

    return sentences_score

'''
    Add an aditional score for the first two and last two proposition as usually
    in the beginning and the ending of a text, the most important information
    is.
'''
def sentence_position_score(sentences_score):
    keys = list(sentences_score.keys())
        
    sentences_score[keys[0]] += most_important
    sentences_score[keys[-1]] += most_important

    if(len(keys) > 4):
        sentences_score[keys[1]] += second_important
        sentences_score[keys[-2]] += second_important

    return sentences_score

'''
    Create the summary of the text.
'''
def create_summary(text, title, tfidf, i):
    #Create the initial dictionary to store the sentences
    sentences_score = create_sentences_dictionary(text)

    #Add the score for the TFIDF stored nouns
    sentences_score = sentences_initial_score(sentences_score, tfidf)

    #Add the score for title similarity between sentences and title
    sentences_score = title_similarity_score(sentences_score, title)
    
    #Add the score for sentences positioned in the beginning/ending of text
    sentences_score = sentence_position_score(sentences_score)
    
    #Keep the three best ranked sentences
    keys = list(sentences_score.keys())
    sorted_sentences = list(sorted(sentences_score.items(), key = lambda kv:
        kv[1], reverse = True))[:3]

    #Remove the score and keep only the index
    sorted_sentences = [x for x,y in sorted_sentences]

    #Get the indexes of the sentences inside the full text
    summary_indexes = []
    for sent in sorted_sentences:
        summary_indexes.append(keys.index(sent))
    summary_indexes.sort()

    #Merge the sentences to keep the original order in text
    summary = ''
    for index in summary_indexes:
        sentence = original_corrected_sentences[i][index]
        summary += sentence[0].upper() + sentence[1:] + ' '

    #Remove the space between text and punctuation
    summary = summary.strip()
    summary = re.sub(r'\s+([?.,!"])', r'\1', summary)

    return summary

'''
    Create the n_grams for a specific text.
'''
def compute_n_grams(text, n):
    words = word_tokenize(text)

    #Split the text in ngrams
    n_grams = ngrams(words, n)
    
    dict_grams = {}
    for gram in n_grams:
        tup = tuple(gram)

        #Save the ngram in the dictionary
        if tup not in dict_grams:
            dict_grams[tup] = 1
    
    return dict_grams

'''
    Count the number of overlaps between two sets of ngrams.
'''
def number_overlaps(first, second):
    result = 0

    for gram in first:
        if gram in second:
            result += 1

    return result

'''
    Calculate the ROUGE and BLEU scores.
'''
def rouge_bleu_n(summary, ground_truth, n):
    #Compute the ngrams for the summary and the ground truth
    summary_grams = compute_n_grams(summary, n)
    truth_grams = compute_n_grams(ground_truth, n)

    #Count the number of overlaps between the two
    nr_overlaps = number_overlaps(summary_grams, truth_grams)

    #Calculate ROUGE@N and BLEU@N
    rouge_n = nr_overlaps / len(truth_grams)
    bleu_n = nr_overlaps / len(summary_grams)

    return (rouge_n, bleu_n)

'''
    Evaluate the generation of a summary by computing the
    BLEU_1, BLEU_2, BLEU_4, ROUGE_1, ROUGE_2, ROUGE_4 scores.
'''
def evaluate_summary(summary, ground_truth):
   
    (rouge_1, bleu_1) = rouge_bleu_n(summary, ground_truth, 1)
    (rouge_2, bleu_2) = rouge_bleu_n(summary, ground_truth, 2)
    (rouge_4, bleu_4) = rouge_bleu_n(summary, ground_truth, 4)

    return (rouge_1, rouge_2, rouge_4, bleu_1, bleu_2, bleu_4)

def main():
    global original_corrected_sentences

    #Dictionary to store the data from the file
    components = {}

    #Read the input file
    read_dataset(components)
    nr_textes = len(components[headline_tag])
  
    TFs = []

    for i in range(nr_textes):
        #Check that text is not empty
        if len(components[ctext_tag][i]) > 0:
            #Clean the text and compute the TF
            components[ctext_tag][i] = clean_text(components[ctext_tag][i],
                    TFs)
            print(i)

        else:
            #For empty texts append empty data to keep the order
            TFs.append({})
            original_corrected_sentences.append({})

    #Compute the TFIDF
    TFIDF = build_tfidf(TFs, nr_textes)

    #Final values for the metrics
    data_rouge_1 = 0
    data_rouge_2 = 0
    data_rouge_4 = 0
    data_bleu_1 = 0
    data_bleu_2 = 0
    data_bleu_4 = 0

    not_empty_texts = 0
    
    for i in range(nr_textes):
        if len(components[ctext_tag][i]) > min_size:
           
            #Create the summary for the text
            summary = create_summary(components[ctext_tag][i],
                    components[headline_tag][i], TFIDF[i], i)

            print(summary)
            #Compute the metrics for the summary
            (rouge_1, rouge_2, rouge_4, bleu_1, bleu_2, bleu_4) = evaluate_summary(summary, components[text_tag][i])
            
            #Add the result to the final one
            data_bleu_1 += bleu_1
            data_bleu_2 += bleu_2
            data_bleu_4 += bleu_4

            data_rouge_1 += rouge_1
            data_rouge_2 += rouge_2
            data_rouge_4 += rouge_4
            
            not_empty_texts += 1

    #Compute the result for the entire dataset
    print(data_bleu_1 / not_empty_texts)
    print(data_bleu_2 / not_empty_texts)
    print(data_bleu_4 / not_empty_texts)

    print(data_rouge_1 / not_empty_texts)
    print(data_rouge_2 / not_empty_texts)
    print(data_rouge_4 / not_empty_texts)
    
if __name__ == "__main__":
    main()
