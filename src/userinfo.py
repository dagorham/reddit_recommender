from collections import Counter

import praw
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from build_reddit import Reddit, Subreddit
from wordcloud import WordCloud

__name__ = 'userinfo'


class Redditor:
    def __init__(self, username):
        self.username = username

        # check if the user exists
        try:
            r = praw.Reddit(user_agent=self.username + '_comment_scraper')
            test = r.get_redditor(self.username).fullname
            self.sub_ratings, self.sub_objs, self.comments = self.get_favorite_subs()
            self.num_subs = len(self.sub_objs)
            self.vectorizer = TfidfVectorizer()
            self.comments = np.array([self.comments])
            self.comments_vectorized = self.vectorizer.fit_transform(self.comments)

        except praw.errors.NotFound:
            self.username = None

    def get_favorite_subs(self):
        """
        :return:

        sub_ratings: dictionary where keys are subreddits, values are percentage of
        the user's posts in that subreddit

        sub_odjs: list of subreddit objects for the subs the user is subscribed to

        comments: string of user comments for word cloud use
        """

        r = praw.Reddit(user_agent=self.username + '_comment_scraper')
        user = r.get_redditor(self.username)

        subs = []
        sub_objs = []
        comments = ''
        sub_ratings = {}

        for i, comment in enumerate(user.get_comments(limit=None)):
            print("Getting comment {}.".format(i))

            subs.append(str(comment.subreddit))
            comment_text = comment.body

            comments += comment_text

        print("Counting posts...")

        post_count = Counter(subs)  # get post count for each sub

        total_posts = sum(post_count.values())  # total posts by user

        print("Converting counts to frequencies...")

        for key, value in post_count.items():
            sub_ratings[key] = 1.0*value/total_posts  # convert number of posts to percent of total
            try:
                sub_objs.append(Subreddit(key))

            except KeyError:
                continue

        print("Done")

        return sub_ratings, sub_objs, comments

    def generate_recommendation(self):
        """
        :return: list of reddits in order of recommendation score
        """

        rec_arr = []

        for sub in self.sub_objs:
            print("\nGetting similar subs for subreddit {}".format(sub.name))

            similarities = list(sub.get_similar_subreddits())
            # get row from commonality matrix

            rec_arr.append(similarities)

        rec_arr = np.array(rec_arr)

        for sub in self.sub_objs:
            # set all rows corresponding to already subscribed subreddits to zero
            rec_arr[:, sub.index] = np.zeros(self.num_subs)

        print("Getting average scores...\n")

        # sum down the columns and average
        final_rec_vector = 1/self.num_subs * np.apply_along_axis(sum, 0, rec_arr)

        return [Reddit.index_to_sub[i] for i in np.argsort(-final_rec_vector)]

    def print_recommendations(self):
        """
        print out recommended subs 10 at a time
        :return: void
        """

        rec_vector = self.generate_recommendation()

        print("Recommendations for user {} ".format(self.username))

        for ranking, subreddit_name in enumerate(rec_vector, 1):
            print("{}.: {}".format(ranking, subreddit_name))

            if ranking%10 == 0 and ranking!=0:
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

    @classmethod
    def print_recommendations_from_user_input(self):
        """
        takes username as input and prints recommended subs
        :return: void
        """

        getting_name = True

        print("Please enter username and press enter:\n")

        while getting_name:
            username = input()

            redditor = self(username)

            if not redditor.username:
                print("Redditor does not exist. Please enter again.\n")
                continue

            break

        redditor.print_recommendations()

    def show_word_cloud(self):
        """
        generates a word cloud of the user's comments
        :return: void
        """

        cloud = WordCloud(str(self.comments))
        cloud.show_cloud()

    @classmethod
    def word_cloud_from_user_input(self):
        """
        asks user for their username
        scrapes their comments
        generates a wordcloud
        :return: void
        """

        getting_name = True

        print("Please enter username and press enter:\n")

        while getting_name:
            username = input()

            redditor = self(username)

            if not redditor.username:
                print("Redditor does not exist. Please enter again.\n")
                continue

            break

        redditor.show_word_cloud()

    def print_topics(self, num_topics=10, num_words=10):
        """
        prints 10 topics based on LDA
        :return: void
        """

        topic_modeler = LatentDirichletAllocation(n_topics=num_topics, learning_method='online')

        topic_modeler.fit(self.comments_vectorized)

        word_list = self.vectorizer.get_feature_names()

        for topic_number, topic in enumerate(topic_modeler.components_):
            top_ten = np.argsort(-topic)[:num_words]

            words_ranked = ", ".join([word_list[i] for i in top_ten])

            print("Topic {}: {}".format(topic_number, words_ranked))

    @classmethod
    def topics_from_user_input(self):
        """
        takes username as input
        scrapes for comments and generates topic model
        prints them out
        :return: void
        """

        getting_name = True

        print("\nPlease enter username and press enter:\n")

        while getting_name:
            username = input()

            redditor = self(username)

            if not redditor.username:
                print("Redditor does not exist. Please enter again.\n")
                continue

            break

        redditor.print_topics()
