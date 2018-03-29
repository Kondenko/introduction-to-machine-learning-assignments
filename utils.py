import os


def get_project_root():
    path = os.path.abspath(__file__)
    return os.path.dirname(path)


def get_datasets_folder():
    return os.path.join(get_project_root(), "datasets")


def get_csv_path(name):
    return "{}/{}.csv".format(get_datasets_folder(), name)


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

    def __init__(self):
        pass

    @staticmethod
    def print_title(title):
        print "\n||| " + title + " |||\n"

    @staticmethod
    def write_to_file(name, text):
        import errno
        try:
            answers_dir = "answers"
            try:
                os.makedirs(answers_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            answer_file = "{}\\{}.txt".format(answers_dir, str(name))
            with open(answer_file, "w") as f:
                f.write(str(text))
        except IOError:
            pass

    def print_answer(self, title, answer, write_to_file = True):
        self.print_title(title)
        print answer
        if write_to_file:
            self.write_to_file(title, answer)

    def execute(self, title, algorithm):
        self.print_title(title)
        answer = algorithm()
        print answer
        self.write_to_file(title, answer)
