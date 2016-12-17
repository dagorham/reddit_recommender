import random
import heapq
from collections import Counter
from string import punctuation
from math import sqrt
from PIL import Image, ImageDraw, ImageFont

from nltk.corpus import stopwords
import numpy as np

__name__ = 'wordcloud'


class WordCloud:
    def __init__(self, text, verbose=False, preprocessed=False):
        self.text = text
        self.verbose = verbose
        self.preprocessed = preprocessed
        self.cloud = None

    def show_cloud(self):
        """
        show the word cloud
        :return: void
        """

        if not self.cloud:
            self.generate_cloud()

        self.cloud.show()

    def generate_cloud(self):
        """
        generate the word cloud
        :return: void
        """

        # process the text if it hasn't been already
        if not self.preprocessed:
            stopwords_text = stopwords.words('english')
            processed = self._process(self.text, stopwords_text)

        else:
            processed = self.text

        counts_first = Counter(processed)  # counts of all words

        # a test image is necessary to compute the size of each word we need
        test = Image.new('RGBA', (470, 350), (255, 255, 255, 255))
        check = ImageDraw.Draw(test)

        # size of the largest word
        master_font = 300

        # total area needed to fit all the words
        total_area_needed = 0

        # saving all the fonts, sizes, and words, in the same order
        fonts = []
        sizes = []
        words = []

        word_heap = []

        # put all the words in a priority queue so i can pull them out in order of size
        for word, count in counts_first.items():
            heapq.heappush(word_heap, (-count, word))

        if self.verbose:
            print("Sizing canvas...", end="")

        rank = 1
        total_words = len(word_heap)

        # only doing 25 words, might adjust this later...
        while rank <= total_words and rank <= 25:
            count, word = heapq.heappop(word_heap)

            # font size inverse the word's rank
            font = ImageFont.truetype('Users/family/Library/Fonts/Arial Bold.ttf', size=int(1/rank * master_font))

            size = check.textsize(word, font=font)  # size of the rectangle necessary for the word

            total_area_needed += size[0] * size[1]

            fonts.append(font)
            sizes.append(size)
            words.append(word)

            rank += 1

        if self.verbose:
            print("Done\n")

        # total canvas size a little bigger than all the words added together
        height = int(1.5 * sqrt(total_area_needed / 2))
        width = int(1.5 * 2 * sqrt(total_area_needed / 2))

        # new cloud with calc'd size
        wrdcld = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        d = ImageDraw.Draw(wrdcld)

        # canvas of ones means every pixel starts out available
        canvas = np.ones((height, width))

        # calculate the integral image
        integral_image = np.cumsum(np.cumsum(canvas, axis=0), axis=1)

        if self.verbose:
            print("Locating words...", end="")

        # run thru all the words
        for idx, word in enumerate(words):
            print("Locating word {}".format(idx))

            # get location
            v_loc, h_loc = self._get_index(integral_image, sizes[idx])

            d.text((h_loc, v_loc), word, font=fonts[idx], fill=(0, 0, 0, 0))

            # mark the chosen area as unusable
            canvas[v_loc:v_loc + sizes[idx][1], h_loc:h_loc + sizes[idx][0]] = 0

            # recalc the integral image based on the new canvas
            integral_image = np.cumsum(np.cumsum(canvas, axis=0), axis=1)

        if self.verbose:
            print("Done.\n")

        self.cloud = wrdcld

    def _rectangle_sum(self, upper_left, int_img, img_size):
        """
        :param upper_left: point where you want to put the upper left corner of the word
        :param int_img: integral image matrix showing what parts of the image are taken up
        :param img_size: size of the image
        :return: sum of the rectangle created by placing an image of size img_size, with its
        top left corner being located at upper_left, on the current integral image matrix
        """

        i_a = int_img[upper_left[0], upper_left[1]]
        i_b = int_img[upper_left[0], upper_left[1] + img_size[0]]
        i_c = int_img[upper_left[0] + img_size[1], upper_left[1]]
        i_d = int_img[upper_left[0] + img_size[1], upper_left[1] + img_size[0]]

        return i_d - i_b - i_c + i_a

    def _get_index(self, int_img, img_size):
        """
        :param int_img: integral image matrix
        :param img_size: size of the image to be placed
        :return: point on the canvas to place the next word, chosen at random from all viable points
        """

        # empty list of legal indices
        possible_indices = []

        img_area = img_size[0] * img_size[1]

        # check all point on canvas
        for i in range(0, int_img.shape[0] - img_size[1] - 1):
            for j in range(0, int_img.shape[1] - img_size[0] - 1):

                # calc available space for point i, j
                sum_at_this_point = self._rectangle_sum((i, j), int_img, img_size)

                if sum_at_this_point >= img_area:
                    possible_indices.append((i, j))

        return random.choice(possible_indices)

    def _process(self, string_in, stopwords):
        """
        :param string_in: string to process
        :param stopwords: list of stopwords to remove
        :return: processed string
        """

        string_in = string_in.lower()

        # remove punction
        for char in string_in:
            if char in punctuation:
                string_in = string_in.replace(char, "")

            try:
                int(char)
                string_in = string_in.replace(char, "")

            except:
                continue

        # get rid of stopwords
        return [x for x in string_in.split() if x not in stopwords]
