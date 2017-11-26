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
            path = """F:\\Python projects\\projects\\IntroToMachineLearning\\{}\\answers\\{}.txt""" \
                .format(str(name), self.assignment)
            f = open(path, "w")
            try:
                f.write(str(text))
            finally:
                f.close()
        except IOError:
            pass

    def execute(self, title, algorithm):
        self.print_title(title)
        answer = algorithm()
        print answer
        self.write_to_file(title, answer)
