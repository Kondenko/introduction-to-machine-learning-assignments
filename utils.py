import os


def get_project_root():
    path = os.path.abspath(__file__)
    return os.path.dirname(path)


def get_datasets_folder():
    return get_project_root() + "\\datasets\\"


def get_titanic_dataset(pandas):
    return pandas.read_csv(
        filepath_or_buffer=(get_datasets_folder() + "titanic\\titanic.csv"),
        index_col='PassengerId',
        engine='python'
    )


def percent(part, total):
    return part * 100.0 / total


def round2(number):
    return round(number, 2)


class Executor:

    assignment = ''

    def __init__(self, assignment_name):
        self.assignment = assignment_name

    @staticmethod
    def print_title(title):
        print "\n||| " + title + " |||\n"

    def write_to_file(self, name, text):
        try:
            path = """{}\\{}\\answers\\{}.txt""" \
                .format(get_project_root(), self.assignment, str(name))
            # print "Writing to {}".format(path)
            f = open(path, "w")
            try:
                f.write(str(text))
            finally:
                f.close()
        except IOError:
            pass

    def print_answer(self, title, answer):
        self.print_title(title)
        print answer
        self.write_to_file(title, answer)

    def execute(self, title, algorithm):
        self.print_title(title)
        answer = algorithm()
        print answer
        self.write_to_file(title, answer)
