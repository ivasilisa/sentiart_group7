import pandas as pd
import nltk
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy import stats


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
    tokens_raw = tokens
    
    #Additionally: parts of speech selection

    #selecting adjectives, adjective comparative, adjectives superlative, nouns, plural nouns, verbs present tense, verbs past tense, gerund, modal verbs, verbs, adverbs
    #list_to_select =  ['JJ','JJR', 'JJS', 'NN', 'NNS', 'VBP', 'VBD', 'VBG', 'MD', 'VB', 'RB'] 
    #list_to_select =  ['JJ', 'NN', 'RB', 'VBG']
    # identifying the part of speech of each word in a line 
    #tokens = [[word[0] for word in nltk.pos_tag(line) if word[1] in list_to_select] for line in tokens] 

    
    return tokens_raw, tokens


def make_sent_means(senti_art, song_tokens):
    """
    estimates mean sentiart values
    
    senti_art: pandas DataFrame, dictionary of 250k english words
    song_tokens: list of lists, list of words in the songs
    
    """

    sent_means = [] #just creating an empty list to use it in a function
    sent_labels = senti_art.columns[1:].tolist() #making a list of column names
    
    #finding words in our tokenized songs
    
    ward = pd.read_csv('WRAD.txt', sep=" ", header=None)
    ward.columns = ['word', 'ward']
    sent_labels.append('WARD')

    valence = pd.read_csv('Valence(j-r).txt', sep=" ", header=None)
    valence.columns = ['word', 'valence']
    sent_labels.append('valnce')
    
    sent_means = np.zeros((len(song_tokens), 9))
    
    for i, t in enumerate(song_tokens):
        dt = senti_art.query('word in @t')
        dt_ward = ward.query('word in @t')
        dt_valence = valence.query('word in @t')
        
        #cleaning (taking into analysis words that are longer than 2 symbols)
        dt = dt.loc[[True if len(i)>0 else False for i in dt["word"].tolist()], :]
        #estimating the mean for all columns, leaving only numbers and appending them to the empty list created before
        sent_means[i, :7] = dt.iloc[:, 1:].mean().to_numpy().flatten()
        sent_means[i, 7] = dt_ward.mean().to_numpy()[0]
        sent_means[i, 8] = dt_valence.mean().to_numpy()[0]
        
    #changing the type of data: from list to array
    # sent_means = np.array(sent_means)
    
    #making a final data frame
    result = pd.DataFrame(data=sent_means, columns=sent_labels).fillna(0)
    return result


def art_plots(results, query_value_inx, save_path, sheet, df_liking, df_striking):
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

    #plot AAPz
    results.set_index(results.index+1,inplace=True)
    #create new columns with liking and striking mean values
    results['Liking'] = df_liking.mean()
    results['Striking'] = df_striking.mean()
    
    results.plot(kind='bar',alpha=0.75, rot=0, ax=ax)
    plt.xlabel("Sentence #")
    plt.ylabel("Sentiment Value (z)")
    
    file_name = f"song_{sheet}_{value_name}.png" 
    plt.savefig(fname=save_path.parent / file_name, dpi=200)
    plt.close()
    


    
def full_processing(song_file, sheet, sa, df_liking, df_striking, only_song_results=False):
    """
    
    
    """
    song_file = Path(song_file)
    
    if song_file.is_file():
        
        # Step 1
        tokens_raw, tokens = tokenize_song(song_file, sheet)

        # Step 2
        song_results = make_sent_means(sa, tokens)
        
        if only_song_results:
            return tokens_raw, tokens, song_results

        # Step 3
        # Select only AAPz
        for i in [0]: # range(len(song_results.columns))
            art_plots(song_results, i, song_file, sheet, df_liking, df_striking)


        # Step 4
        # additional_stats(sa, song_results)
        
        # Step 5 - save results
        song_results.to_excel(song_file.parent / f"song_{sheet}.xlsx")
        
        # Step 6
        bag_of_words = list(set(sum(tokens_raw, [])))
        values = sa.query("word in @bag_of_words")
        values.to_excel(song_file.parent / f"song_words_list_{sheet}.xlsx")
        
        return [bag_of_words, values]
        #return [len(bag_of_words), values.shape[0]]
        print('DONE')
    else:
        print('The file does not exist')
        
        
def normalize(df):
    data =  df.to_numpy()
    data_std = (data - data.mean(axis=1, keepdims=True))/data.std(axis=1, keepdims=True) 
    return pd.DataFrame(data=data_std)


def plot_norm_outliers(df, song, group, n_not_norm=3):
    for i, x in enumerate(df.to_numpy()):
        _, p = stats.kstest(x, 'norm')
        df.loc[i, 'is_norm'] = p

    sort_by_norm = df['is_norm'].to_numpy().argsort()

    fig, ax = plt.subplots(figsize=(10, 5))
    df.iloc[:, :-1].T.plot.kde(legend=False, ax=ax)
    df.iloc[sort_by_norm[n_not_norm:], :-1].T.plot.kde(legend=False, ax=ax, c='grey')
    df.iloc[sort_by_norm[:n_not_norm], :-1].T.plot.kde(legend=False, ax=ax, c='red')

    plt.xlabel('Standartized responses')
    
    file_name = f"song_{song}_{group}.png"
    plt.show()
    plt.savefig(fname=file_name, dpi=200)
    plt.close()