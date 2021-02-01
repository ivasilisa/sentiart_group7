import pandas as pd
import nltk
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def tokenize_song(path_to_song, sheet):
    """
    tokenize_song: breaks sentences into words
    path_to_song: str, name of the file with songs
    sheet: number of an excel sheet of a song
    
    NOTES
    *there are some not completely tokenized elements, e.g. contractions ('t, 'd)

    """
    song = pd.read_excel(path_to_song, sheet_name=sheet) #access a respective excel file and a sheet
    song_lines = song.loc[:, 'Line'].tolist() #access only one column using label "Line" 


    tokens = [[t for t in nltk.word_tokenize(s) if t.isalpha()] for s in song_lines] #tokenizing
    tokens = [[word.lower() for word in token] for token in tokens] #from upper to lower case (words containing upper-case letters are not analyzed)
    
    #Additionally: parts of speech selection

    #selecting adjectives, nouns, plural nouns, verbs present tense, verbs past tense, modal verbs, verbs, adverbs
    #list_to_select =  ['JJ', 'NN', 'NNS', 'VBP', 'VBD', 'MD', 'VB', 'RB'] 
    #identifying the part of speech of each word in a line 
    #tokens = [[word[0] for word in nltk.pos_tag(line) if word[1] in list_to_select] for line in tokens] 
    
    
    return tokens


def make_sent_means(senti_art, song_tokens):
    """
    estimates mean sentiart values
    
    senti_art: pandas DataFrame, dictionary of 250k english words
    song_tokens: list of lists, list of words in the songs
    
    """

    sent_means = [] #just creating an empty list to use it in a function
    sent_labels = senti_art.columns[1:].tolist() #making a list of column names
    
    #finding words in our tokenized songs
    for t in song_tokens:
        dt = senti_art.query('word in @t')
        
        #cleaning (taking into analysis words that are longer than 2 symbols)
        #dt = dt.loc[[True if len(i)>2 else False for i in dt["word"].tolist()], :]
        #estimating the mean for all columns, leaving only numbers and appending them to the empty list created before
        sent_means.append(dt.iloc[:, 1:].mean().to_numpy())
    
    #changing the type of data: from list to array
    sent_means = np.array(sent_means)
    #making a final data frame
    result = pd.DataFrame(data=sent_means, columns=sent_labels)
    return result


def art_plots(results, query_value_inx, save_path, sheet):
    """
    makes a plot
    
    results: data frame of results
    query_value_inx: int, index to select values in the dataframe of results
    save_path: pathlib.PosixPath, path where to save the data 
    sheet: giving a number to the saved file according to the number of excel sheet in the initial document
    
    """
    
    fig, ax = plt.subplots(figsize=(15, 10)) 
    results = round(results,3)
    value_name = results.columns[query_value_inx]
    results = results.loc[:, [value_name]]
    #results.to_csv('results.txt')

    #plot AAPz, fear_z etc.
    results.set_index(results.index+1,inplace=True)
    
    results.plot(kind='bar',alpha=0.75, rot=0, ax=ax)
    plt.xlabel("Sentence #")
    plt.ylabel("Sentiment Value (z)")
    
    file_name = f"song_{sheet}_{value_name}.png" 
    plt.savefig(fname=save_path.parent / file_name, dpi=200)
    plt.close()

    
def full_processing(song_file, sheet, sa):
    """
    
    
    """
    song_file = Path(song_file)
    
    if song_file.is_file():
        
        # Step 1
        tokens = tokenize_song(song_file, sheet)

        # Step 2
        song_results = make_sent_means(sa, tokens)

        # Step 3
        # Select only AAPz
        for i in [0]: # range(len(song_results.columns))
            art_plots(song_results, i, song_file, sheet)


        # Step 4
        # additional_stats(sa, song_results)
        
        # Step 5 - save results
        song_results.to_excel(song_file.parent / f"song_{sheet}.xlsx")
        
        # Step 6
        
        bag_of_words = sum(tokens, [])
        values = sa.query("word in @bag_of_words")
        values.to_excel(song_file.parent / f"song_words_list_{sheet}.xlsx")
        print('DONE')
    else:
        print('The file does not exist')