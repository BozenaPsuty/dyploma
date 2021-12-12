
from nltk.sentiment.vader import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()

from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from spacy.lang.en import English
nlp = English()  # just the language with no model
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)
import os
import pandas as pd
from string import punctuation
import re
from textblob import TextBlob, Word
import ast
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet


class Processing:

    def remove_punctuation(self, text):
        final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "'", ",",
                                                     "~", "$", "â", "€", "™", "+", "(", ")", "/", "*", "@", "-", "&",
                                                     "’", "‘", "“", "”", "%", "[", "]"))
        return final

    def remove_StopWords(self, rev):
        all_stopwords = stopwords.words('english')
        text_tokens = word_tokenize(rev)
        tokens_without_sw = " ".join(word for word in text_tokens if not word in all_stopwords)
        final = ("").join(tokens_without_sw)
        return final

    def remove_digits(self, text):
        final = "".join(i for i in text if not i.isdigit())
        return final

    def remove_WhiteSpace(self, text):
        final = " ".join(text.split())
        return final

    def split_into_sentences(self, directory):
        i = 0
        for entry in os.scandir(directory):
            if entry.path.endswith(".csv"):
                i = i+1
                print(i)
                df = pd.read_csv(entry)
                df['text'] = df['text'].apply(lambda x: [sent.text for sent in nlp(x).sents])
                test = pd.DataFrame(df['text'].tolist())
                final = pd.concat([df, test], axis=1)
                final.to_csv(directory + 'review' + str(i) + "_new.csv", index=None)

    def process(self, directory, new_path):
        i = -1
        for entry in os.scandir(directory):
            if entry.path.endswith(".csv"):
                i = i + 1
                df = pd.read_csv(entry)
                new_sentences = self.split_into_sentences(df['reviews'])

                df.to_csv(directory + 'csv_new/review' + str(i) + "_new.csv", index=None)

    def clean_sentence(self, sentence):
        sentence = re.sub(r"(?:\@|https?\://)\S+|\n+", "", sentence.lower())
        # Fix spelling errors in comments!
        sent = TextBlob(sentence)
        # sent.correct()
        clean = ""
        for sentence in sent.sentences:
            words = sentence.words
            # Remove punctuations
            words = [''.join(c for c in s if c not in punctuation) for s in words]
            words = [s for s in words if s]
            clean += " ".join(words)
            clean += ". "
        return clean

    def each_clean(self, file_path):
        a = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                text = {}
                i = 0
                for x in df['text']:
                    text[i] = self.clean_sentence(x)
                    i += 1
                text = pd.DataFrame(text, index=[0])
                text = text.T
                df['text'] = text
                print(a)
                df.to_csv(file_path + 'review' + str(a) + "_clean.csv", index=None)
                a += 1

    def tokenizeReviews(self, file):
        tokenizedReviews = {}
        tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
        uniqueId = 1
        for sentence in tokenizer.tokenize(file):
            sentence = self.remove_StopWords(sentence)
            tokenizedReviews[uniqueId] = sentence
            uniqueId += 1
        return(str(tokenizedReviews))

    def tokenize_all(self, file_path):
        a = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                text = {}
                i = 0
                for x in df['text']:
                    text[i] = self.tokenizeReviews(x)
                    i += 1
                text = pd.DataFrame(text, index=[0])
                text = text.T
                df['text2'] = text
                print(a)
                df.to_csv(file_path + 'review' + str(a) + "_clean.csv", index=None)
                a += 1

    def posTagging(self, file):
        output_post = {}
        file = file.split('.')
        a = 0
        for value in file:
            output_post[a] = nltk.pos_tag(nltk.word_tokenize(value))
            a += 1
        return str(output_post)

    def tag_all(self, file_path):
        a = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                text = {}
                i = 0
                for x in df['text']:
                    text[i] = self.posTagging(x)
                    i += 1
                text = pd.DataFrame(text, index=[0])
                text = text.T
                df['text_tagged'] = text
                print(a)
                df.to_csv(file_path + 'new/review' + str(a) + "_clean.csv", index=None)
                a += 1

    def aspectExtraction(self, file):
        inputTupples = ast.literal_eval(file)
        prevWord = ''
        prevTag = ''
        currWord = ''
        aspectList = []
        outputDict = {}
        # Extracting Aspects
        for key, value in inputTupples.items():
            for word, tag in value:
                if (tag == 'NN' or tag == 'NNP'):
                    if (prevTag == 'NN' or prevTag == 'NNP'):
                        currWord = prevWord + ' ' + word
                    else:
                        aspectList.append(prevWord.upper())
                        currWord = word

                prevWord = currWord
                prevTag = tag
        # Eliminating aspect which has 1 or less count
        """for aspect in aspectList:
            if (aspectList.count(aspect) > 1):
                if (outputDict.keys() != aspect):
                    outputDict[aspect] = aspectList.count(aspect)"""
        for aspect in aspectList:
            outputDict[aspect] = aspectList.count(aspect)
        outputAspect = sorted(outputDict.items(), key=lambda x: x[1], reverse=True)
        return str(outputAspect)

    def mine_all_aspects(self, file_path):
        a = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                text = {}
                i = 0
                for x in df['text_tagged']:
                    text[i] = self.aspectExtraction(x)
                    i += 1
                text = pd.DataFrame(text, index=[0])
                text = text.T
                df['aspects'] = text
                print(a)
                df.to_csv(file_path + 'review' + str(a) + "_clean.csv", index=None)
                a += 1

    def lammetization(self, file):
        wordnet_lemmatizer = WordNetLemmatizer()
        inputTupples = ast.literal_eval(file)
        outputPost = {}
        for key, value in inputTupples.items():
            outputPost[key] = [wordnet_lemmatizer.lemmatize(word) for word in nltk.word_tokenize(value)]
            outputPost[key] = ' '.join(outputPost[key])
        return str(outputPost)

    def lamme_all(self, file_path):
        a = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                text = {}
                i = 0
                for x in df['text2']:
                    text[i] = self.lammetization(x)
                    i += 1
                text = pd.DataFrame(text, index=[0])
                text = text.T
                df['text3'] = text
                print(a)
                df.to_csv(file_path + 'review' + str(a) + "_clean.csv", index=None)
                a += 1

    def getSubjectivity(self, review):
        inputTupples = ast.literal_eval(review)
        outputPost = {}
        for key, value in inputTupples.items():
            outputPost[key] = TextBlob(review).sentiment.subjectivity
        return str(outputPost)
        # function to calculate polarity

    def getPolarity(self, review):
        inputTupples = ast.literal_eval(review)
        outputPost = {}
        for key, value in inputTupples.items():
            outputPost[key] = TextBlob(review).sentiment.polarity
        return str(outputPost)

    # function to analyze the reviews
    def analysis(self, score):
        inputTupples = ast.literal_eval(score)
        outputPost = {}
        for key, value in inputTupples.items():
            if value < 0:
                outputPost[key] = 'Negative'
            elif value == 0:
                outputPost[key] = 'Neutral'
            else:
                outputPost[key] = 'Positive'
        return str(outputPost)

    def sentiment(self, file_path):
        a = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                df['subjectivity'] = df['text2'].apply(self.getSubjectivity)
                df['polarity'] = df['text2'].apply(self.getPolarity)
                df['analysis'] = df['polarity'].apply(self.analysis)
                df.to_csv(file_path + 'review' + str(a) + "_clean.csv", index=None)
            print(a)
            a += 1

    def orientation(self, inputWord):
        wordSynset = wordnet.synsets(inputWord)
        if (len(wordSynset) != 0):
            word = wordSynset[0].name()
            orientation = sentiwordnet.senti_synset(word)
            if (orientation.pos_score() > orientation.neg_score()):
                return True
            elif (orientation.pos_score() < orientation.neg_score()):
                return False

    def identifyOpinionWords(self, inputReviewList, inputAspectList):
        inputReviewsTuples = ast.literal_eval(inputReviewList)
        inputAspectTuples = ast.literal_eval(inputAspectList)
        outputAspectOpinionTuples = {}
        orientationCache = {}
        negativeWordSet = open('C:/Users/Bosia/Desktop/negative_words.txt', 'r')

        for aspect in inputAspectTuples:
            aspectTokens = word_tokenize(aspect)
            count = 0
            for key, value in inputReviewsTuples.items():
                condition = True
                isNegativeSen = False
                for subWord in aspectTokens:
                    if (subWord in str(value).upper()):
                        condition = condition and True
                    else:
                        condition = condition and False

                if (condition):
                    for negWord in negativeWordSet:
                        if (
                        not isNegativeSen):  # once senetence is negative no need to check this condition again and again
                            if negWord.upper() in str(value).upper():
                                isNegativeSen = isNegativeSen or True

                    outputAspectOpinionTuples.setdefault(aspect, [0, 0, 0])
                    for word, tag in value:

                        if (tag == 'JJ' or tag == 'JJR' or tag == 'JJS' or tag == 'RB' or tag == 'RBR' or tag == 'RBS'):
                            count += 1
                            if (word not in orientationCache):
                                orien = self.orientation(word)
                                orientationCache[word] = orien
                            else:
                                orien = orientationCache[word]
                            if (isNegativeSen and orien is not None):
                                orien = not orien
                            if (orien == True):
                                outputAspectOpinionTuples[aspect][0] += 1
                            elif (orien == False):
                                outputAspectOpinionTuples[aspect][1] += 1
                            elif (orien is None):
                                outputAspectOpinionTuples[aspect][2] += 1
            if (count > 0):
                # print(aspect,' ', outputAspectOpinionTuples[aspect][0], ' ',outputAspectOpinionTuples[aspect][1], ' ',outputAspectOpinionTuples[aspect][2])
                outputAspectOpinionTuples[aspect][0] = round((outputAspectOpinionTuples[aspect][0] / count) * 100, 2)
                outputAspectOpinionTuples[aspect][1] = round((outputAspectOpinionTuples[aspect][1] / count) * 100, 2)
                outputAspectOpinionTuples[aspect][2] = round((outputAspectOpinionTuples[aspect][2] / count) * 100, 2)
                '''print(aspect, ':\t\tPositive => ', outputAspectOpinionTuples[aspect][0], '\tNegative => ',
                      outputAspectOpinionTuples[aspect][1])'''
        return str(outputAspectOpinionTuples)

    def aspect_sent_all(self, file_path):
        a = 239
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                text = {}
                i = 0
                for x in df['aspects']:
                    text[i] = self.identifyOpinionWords(df['text_tagged'][i], x)
                    i += 1
                text = pd.DataFrame(text, index=[0])
                text = text.T
                df['new'] = text
                df.to_csv(file_path + 'review' + str(a) + "_clean.csv", index=None)
            print(a)
            a += 1
