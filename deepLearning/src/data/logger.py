import sys
import csv
import os


class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.file_name = file_name


    def write(self, message):
        self.terminal.write(message)
        with open(self.file_name, "a") as csv_file:
            csv_file.write(message)


    def flush(self):
        # empty function for python 3 compatibility
        pass

    def revert(self):
        return self.terminal


class CsvWriter:
    def __init__(self, file_path, header, lock=None):
        self.lock = lock
        self.fp = file_path
        self.header = header
        self.write_header()

    def write_header(self):
        if self.lock is not None:
            self.lock.acquire()
        if os.path.isfile(self.fp):
            if self.lock is not None:
                self.lock.release()
            return
        with open(self.fp, 'a') as f:
            writer = csv.DictWriter(f, delimiter=';', lineterminator='\n', fieldnames=self.header, restval=-1)
            writer.writeheader()
        if self.lock is not None:
            self.lock.release()

    def write_row(self, **kwargs):
        if self.lock is not None:
            self.lock.acquire()
        with open(self.fp, 'a') as f:
            writer = csv.DictWriter(f, delimiter=';', lineterminator='\n', fieldnames=self.header, restval=-1)
            writer.writerow(kwargs)
        if self.lock is not None:
            self.lock.release()


if __name__ == '__main__':
    test_fp = '/share/wandell/data/reith/circles_experiment/white_circle_rad_6/test.csv'
    header = ["ene", 'mene', 'mu']
    import multiprocessing as mp
    lock = mp.Lock()
    wrt = CsvWriter(test_fp, header)
    wrt.write_row(ene=2, mene=3, mu=4)
    wrt.write_row(ene=20, mene=30, mix=4)
    print('done')
