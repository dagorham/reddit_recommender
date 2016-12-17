__name__ = 'main'

from build_reddit import Reddit, Subreddit
from userinfo import Redditor


class App:
    quit = False

    def __init__(self, main_menu):
        self.main_menu = main_menu

    def run(self):
        reddit = Reddit()

        # run the app
        while not App.quit:
            self.main_menu.execute_menu()


class Menu:
    def __init__(self, menu_name, option_list, children=None, parent=None, option_children_lineup={},
                 option_function_lineup={}, sub_menu=False):

        self.sub_menu = sub_menu
        self.menu_name = menu_name
        self.option_list = option_list
        self.parent = parent
        self.print_menu = True

        # children will be a list of sub-menus
        self.children = children

        # set the parent of all children to be this object
        if children:
            for child in children:
                child.parent = self

        # option children lineup is a dictionary
        # keys: option numbers in menu
        # values: submenu object corresponding to that option number
        self.option_children_lineup = option_children_lineup
        self.option_function_lineup = option_function_lineup

        if self.sub_menu:
            self.option_list.append("Move up one menu.")

        # defines the option number that will quit out this menu
        self.quit_option = len(self.option_list) + 1

        self.option_list.append("Quit.")

    def execute_menu(self):
        while self.print_menu:
            print(self.menu_name + ":\n")

            for option_number, option in enumerate(self.option_list):
                print(str(option_number + 1) + ") " + option + "\n")

            response = input()

            try:
                response = int(response)

                if response > len(self.option_list):
                    print("This is not a valid response, please try again.\n")
                    continue

            except ValueError:
                print("This is not a valid response, please try again.\n")
                continue

            response = int(response)

            # check if the response is to go to a submenu
            if response in self.option_children_lineup.keys():
                sub_menu = self.children[self.option_children_lineup[response]]

                sub_menu.execute_menu()  # go to the submenu if it is

            if response in self.option_function_lineup:
                self.option_function_lineup[response]()

            if response == self.quit_option - 1:
                break

            # check if response is to quit entirely
            if response == self.quit_option:
                self.print_menu = False

                if self.sub_menu:
                    self.parent.print_menu = False

                App.quit = True

main_options = ['View subreddit info.',\
                'View user info.']

main_option_children_lineup = {1: 0, 2: 1}

user_options = ['Get recommendations.',
                'Get user\'s topics',
                'Create comment word cloud.']

user_option_function_lineup = {1: Redditor.print_recommendations_from_user_input,
                               2: Redditor.topics_from_user_input,
                               3: Redditor.word_cloud_from_user_input}

sub_options = ['View similar subreddits.',
               'Get subreddit\'s topics.',
               'Create comment word cloud.']

sub_option_function_lineup = {1: Subreddit.similar_subs_from_user_input,
                              2: Subreddit.topics_from_user_input,
                              3: Subreddit.word_cloud_from_user_input}

user_menu = Menu("User Options", user_options, sub_menu=True, option_function_lineup=user_option_function_lineup)

sub_menu = Menu("Subreddit Options", sub_options, sub_menu=True, option_function_lineup=sub_option_function_lineup)

main_menu = Menu("Main Menu", main_options, children=[sub_menu, user_menu],
                 option_children_lineup=main_option_children_lineup)


def main():
    print("Welcome to the Reddit Recommender\n")

    print("Initializing...\n")

    myApp = App(main_menu)

    myApp.run()

main()

if __name__ == '__main__':
    main()
