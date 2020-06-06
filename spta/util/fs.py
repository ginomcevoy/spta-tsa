import os


def mkdir(directory):
    '''
    Creates directory, does not complain if dir already exists
    '''

    try:
        os.makedirs(directory)
    except OSError:
        pass
