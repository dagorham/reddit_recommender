import re
import os
import sqlite3
import pickle
from string import punctuation

import praw
import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from nltk.corpus import stopwords

from create_db import create_db
from wordcloud import WordCloud
from text_processor import process_text

__name__ = 'build_reddit'


class Reddit:
    # initialize class variables
    text_matrix = None
    unstemmed_text_matrix = None
    vectorized_text_matrix = None
    text_matrix_reduced = None
    sub_list = None
    sub_to_index = None
    index_to_sub = None
    commonality_matrix = None

    def __init__(self, from_db=True, encoding_type='tfidf', distance_method='cosine'):
        # check if everything already exists
        # get it from the db or from the web if not
        try:
            print("Checking if information is available...", end="")
            Reddit.text_matrix = np.load('text_matrix.npy')
            Reddit.unstemmed_text_matrix = np.load('unstemmed_text_matrix.npy')
            Reddit.sub_list = pickle.load(open("sub_list.p", "rb"))
            Reddit.sub_to_index = pickle.load(open("sub_to_index.p", "rb"))
            print("Done.\n")

        except FileNotFoundError:
            print("Not available.\n")
            print("Loading from database.\n")
            print("This will take a minute...\n")

            if from_db:
                Reddit.text_matrix, Reddit.sub_list, Reddit.sub_to_index = self.data_from_db()

            else:
                Reddit.text_matrix, Reddit.sub_list, Reddit.sub_to_index = self.data_from_scrape()

        Reddit.index_to_sub = {value: key for key, value in Reddit.sub_to_index.items()}

        if encoding_type == 'tfidf':
            self.vectorizer = TfidfVectorizer()

        elif encoding_type == 'count':
            self.vectorizer = CountVectorizer()

        elif encoding_type == 'hash':
            self.vectorizer = HashingVectorizer()

        if distance_method == 'cosine':
            self.distance = self.cosine_distance

        try:
            Reddit.vectorized_text_matrix = np.load('vectorized_text_matrix.npy')
            Reddit.text_matrix_reduced = np.load('text_matrix_reduced.npy')

        except FileNotFoundError:
            print("Vectorizing and reducing text matrix.\n")
            Reddit.vectorized_text_matrix, Reddit.text_matrix_reduced = self.process_text_matrix()
            print("Done\n")

        # check if the commonality matrix already exists, build it if it doesn't
        try:
            Reddit.commonality_matrix = np.load('commonality_matrix.npy')

        except FileNotFoundError:
            Reddit.commonality_matrix = self.build_matrix()

    def data_from_db(self):
        """
        get subreddit corpus from database reddit.db

        :return:
        text_matrix: matrix of text in subreddits. rows are subreddits.
        sub_list: list of subreddits included in the matrix
        sub_to_index: dictionary for converting from subreddit name to index in the matrix
        """

        sub_list = []
        text_matrix = []
        unstemmed_text_matrix = []  # used for word cloud later

        connecting_to_db = True

        sql_command = "SELECT subreddit, GROUP_CONCAT(body, ' ') as all_comments FROM comments GROUP BY subreddit"

        while connecting_to_db:
            try:
                print("Connecting to DB.\n")
                pwd = os.getcwd()
                db_conn = sqlite3.connect(pwd + '/../db/reddit.db')
                c = db_conn.cursor()
                results = c.execute(sql_command)

            except sqlite3.OperationalError:
                print("Table does not exist yet. Creating from CSV.\n")
                create_db(db_conn)
                continue

            print("Done.")

            break

        english_stop_words = stopwords.words('english')

        r = praw.Reddit(user_agent='daniel_scraper')

        for i, row in enumerate(list(results)):
            print("Loading subreddit {}: {}....".format(i, row[0]), end="")

            '''
            try:
                if r.get_subreddit(row[0]).subscribers < 50000:
                    print("Done")
                    continue

            except:
                print("Something went wrong. Continuing.")
                continue
            '''

            sub_list.append(row[0].lower())
            text_matrix.append(process_text(row[1], punctuation, english_stop_words))

            unstemmed_text_matrix.append(process_text(row[1], punctuation, english_stop_words, stem=False))

            print("Done")

        sub_to_index = {sub_name: index for sub_name, index in zip(sub_list, range(len(sub_list)))}

        print("Done.\n")

        text_matrix = np.array(text_matrix)
        unstemmed_text_matrix = np.array(unstemmed_text_matrix)

        np.save('unstemmed_text_matrix.npy', unstemmed_text_matrix)
        np.save('text_matrix.npy', text_matrix)
        pickle.dump(sub_list, open("sub_list.p", "wb"))
        pickle.dump(sub_to_index, open("sub_to_index.p", "wb"))

        return text_matrix, sub_list, sub_to_index

    def data_from_scrape(self):
        """
        get subreddit corpus from web scrape if database is not available

        :return:
        text_matrix: matrix of text in subreddits. rows are subreddits.
        sub_list: list of subreddits included in the matrix.
        sub_to_index: dictionary for converting from subreddit name to index in the matrix.
        """

        text_matrix = []

        response = requests.get('http://redditlist.com/sfw')

        sub_list = re.findall('/r/(\w+)\\\'', response.text)
        sub_list = set(sub_list)

        r = praw.Reddit(user_agent='daniel_scraper')

        for sub in self.sub_list:
            if r.get_subreddit(sub).subscribers < 50000:
                self.sub_list.pop(sub)

        sub_list = list(sub_list)

        for sub in sub_list:
            # instantiate string of submission and comments for this specific subreddit
            this_subs_submissions = ''
            this_subs_comments = ''

            submissions = r.get_subreddit(sub).get_hot(limit=25)  # get the top 25 submissions

            for submission in submissions:
                this_subs_submissions += " "
                this_subs_submissions += submission.title.lower()  # add submission to all submissions

                for comment in submission.comments:
                    this_subs_comments += " "
                    this_subs_comments += comment.body.lower()  # add comment to all comments

            text_matrix.append(this_subs_submissions + this_subs_comments)

        text_matrix = np.array(text_matrix)
        sub_to_index = {sub_name: index for sub_name, index in zip(sub_list, range(len(sub_list)))}

        np.save('text_matrix.npy', text_matrix)

        return text_matrix, sub_list, sub_to_index

    def process_text_matrix(self, n_components=100):
        """
        :param n_components: number of singular values to retain
        :return: reduced dimension text matrix using truncated SVD
        """

        vectorized_text_matrix = self.vectorizer.fit_transform(self.text_matrix)
        reducer = TruncatedSVD(n_components=n_components)
        text_matrix_reduced = reducer.fit_transform(vectorized_text_matrix)

        np.save('vectorized_text_matrix.npy', vectorized_text_matrix)
        np.save('text_matrix_reduced.npy', text_matrix_reduced)

        return vectorized_text_matrix, text_matrix_reduced

    @staticmethod
    def cosine_distance(vec1, vec2):
        """
        :param vec1: 1D numpy array
        :param vec2: 1D numpy array
        :return: cosine distance between the two vectors
        """

        # confirm they're numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        return vec1.dot(vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

    def build_matrix(self):
        """
        :return: Reddit "commonality matrix" C

        C[i,j] corresponds to the similarity between subreddit i and subreddit j
        Distance measure is a parameter to the class, defaults to cosine distance
        """

        # initialize a commonality matrix
        commonality_matrix = np.zeros((Reddit.text_matrix.shape[0], Reddit.text_matrix.shape[0]))

        for i in range(len(commonality_matrix)):
            for j in range(i, len(commonality_matrix)):
                commonality = self.distance(Reddit.text_matrix_reduced[i], Reddit.text_matrix_reduced[j])

                commonality_matrix[i, j] = commonality

            commonality_matrix[(i+1):, i] = commonality_matrix[i, (i+1):]

        # save commonality matrix for later use
        np.save('commonality_matrix.npy', commonality_matrix)

        return commonality_matrix


class Subreddit(Reddit):
    def __init__(self, name):
        self.index = Reddit.sub_to_index[name]
        self.similar_subs = Reddit.commonality_matrix[self.index,:]
        self.name = name
        self.text = np.array([str(Reddit.text_matrix[self.index])])
        self.unstemmed_text = str(Reddit.unstemmed_text_matrix[self.index])

        self.vectorizer = TfidfVectorizer()

        self.text_vectorized = self.vectorizer.fit_transform(self.text)

    def get_similar_subreddits(self):
        """
        :return: row of commonality matrix corresponding to the subreddit
        """

        return Reddit.commonality_matrix[self.index, :]

    def print_similar_subreddits(self):
        """
        prints out the most similar subreddits in groups of 10
        :return: void
        """

        print("\n")
        print("Similar subreddits for " + self.name + ":\n")

        similar_subs = self.get_similar_subreddits()
        sorted_by_index = np.argsort(-similar_subs)
        subs_ranked_and_by_name = [Reddit.index_to_sub[i] for i in sorted_by_index]

        for ranking, sub in enumerate(subs_ranked_and_by_name):
            if ranking == 0:
                continue

            print(str(ranking) + ": " + sub)

            if ranking%10 == 0:
                check_if_move_on = True
                print("\nType c and press enter for the next 10 subreddits.\n")
                print("Type q and press enter to return to main menu.\n")

                while check_if_move_on:
                    choice = input()

                    if choice == 'c':
                        break

                    elif choice == 'q':
                        break

                    else:
                        print("Not a valid entry, please enter again.")

                # break the whole thing if they want to quit
                if choice == 'q':
                    break

        print("\n")

    @classmethod
    def similar_subs_from_user_input(self):
        """
        takes a subreddit name as user input
        prints out the most similar subreddits 10 at a time
        :return: void
        """

        getting_sub_name = True

        while getting_sub_name:
            print("Please enter subreddit name: \n")
            response = input()

            try:
                sub = self(response)
                break

            except KeyError:
                print("\nSubreddit not in database. Please enter another.\n")
                continue

        sub.print_similar_subreddits()

    def print_topics(self, num_topics=10, num_words=10):
        """
        performs a topic modeling on the subreddit using LDA
        :param num_topics: number of topics to split subreddit into
        :param num_words: number of words to associate with each topic
        :return: void
        """

        topic_modeler = LatentDirichletAllocation(n_topics=num_topics, learning_method='online')

        topic_modeler.fit(self.text_vectorized)

        word_list = self.vectorizer.get_feature_names()

        for topic_number, topic in enumerate(topic_modeler.components_):
            top_ten = np.argsort(-topic)[:num_words]

            words_ranked = ", ".join([word_list[i] for i in top_ten])

            print("Topic {}: {}".format(topic_number, words_ranked))

    @classmethod
    def topics_from_user_input(self):
        """
        takes a subreddit name as user input and generates topics
        :return: void
        """

        getting_sub_name = True

        while getting_sub_name:
            print("Please enter subreddit name: \n")
            response = input()

            try:
                sub = self(response)
                break

            except KeyError:
                print("Subreddit not in database. Please enter another.\n")
                continue

        sub.print_topics()

    def show_word_cloud(self):
        """
        generates wordcloud of subreddit's comments
        :return: void
        """

        cloud = WordCloud(str(self.text[0]))
        cloud.show_cloud()

    @classmethod
    def word_cloud_from_user_input(self):
        """
        takes a subreddit name as user input
        generates a word cloud of its comments
        :return: void
        """

        getting_sub_name = True

        while getting_sub_name:
            print("Please enter subreddit name: \n")
            response = input()

            try:
                sub = self(response)
                break

            except KeyError:
                print("Subreddit not in database. Please enter another.\n")
                continue

        sub.show_word_cloud()