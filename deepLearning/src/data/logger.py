import sys

class Logger(object):
    def __init__(self, fileName):
        self.terminal = sys.stdout
        self.file = open(fileName, "a")


    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)


    def flush(self):
        # empty function for python 3 compatibility
        pass

    def revert(self):
        return self.terminal
