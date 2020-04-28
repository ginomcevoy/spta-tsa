# Based on https://www.journaldev.com/15631/python-multiprocessing-examaple

import numpy as np
from multiprocessing import Manager, Process, current_process
import logging
import queue


class Parallel2DTask(object):
    '''
    An object that executes a function using one index to track progress.
    Useful for parallel workloads over a 2D region or a spatio-temporal region.
    The task_func must have the following signature:
    task_func(i, j, row_len, task_argv)
    '''

    def __init__(self, shared_obj, i, j, row_len):
        '''
        The shared_obj is an array that will be shared among tasks to store partial results.
        '''
        self.shared_obj = shared_obj
        self.i = i
        self.j = j
        self.row_len = row_len

    def work(self, task_func, task_argv):
        '''
        Perform the assigned task and store the result using the supplied indices
        '''
        k = self.i * self.row_len + self.j
        self.shared_obj[k] = task_func(self.i, self.j, task_argv)

    def __str__(self):
        return '({}, {})'.format(self.i, self.j)


class Parallel2DQueue(object):
    '''
    A queue that is used to manage the execution of ParallelArrayTask instances over two indices.
    Useful for parallel workloads over a 2D region or a spatio-temporal region.
    '''

    def __init__(self, num_proc):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.num_proc = num_proc
        # self.queue = Queue()
        self.queue = Manager().Queue()

    def prepare_queue(self, shape_2D):

        # assume 2D matrix
        (m, n) = shape_2D
        self.qeueue_len = m * n

        # we will use an underlying 1D array as a shared object for the parallel tasks
        empty_seq = np.zeros(self.qeueue_len)
        self.shared_obj = Manager().Array('d', list(empty_seq))

        # initialize all the mxn tasks
        for k in range(0, self.qeueue_len):

            i = int(k / n)
            j = int(k % n)

            # enqueue a task that knows its place on the 2D region
            task_k = Parallel2DTask(self.shared_obj, i, j, n)
            self.queue.put(task_k)

    def execute_func_on_2D(self, task_func, task_argv, shape_2D):

        self.prepare_queue(shape_2D)

        # create and start the processes that will consume the tasks
        processes = []
        for proc_index in range(0, self.num_proc):
            proc = Process(target=process_one_task, args=(self.queue, task_func, task_argv,
                                                          self.logger))
            processes.append(proc)
            proc.start()

        # wait for completion
        for proc in processes:
            proc.join()

        # pull the results from the shared object and create a 2D output
        result = np.array(self.shared_obj[:])
        result = result.reshape(shape_2D)

        return result


def process_one_task(task_queue, task_func, task_argv, logger):
    '''
    Based on https://www.journaldev.com/15631/python-multiprocessing-examaple
    '''

    while True:
        try:
            '''
            try to get task from the queue. get_nowait() function will
            raise queue.Empty exception if the queue is empty.
            queue(False) function would do the same task also.
            '''
            parallel_2D_task = task_queue.get_nowait()

        except queue.Empty:
            # nothing left on queue, this processes has finished
            break

        else:
            '''
            if no exception has been raised, we have a task, work on it
            '''
            proc_name = current_process().name
            logger.debug('Process {} working on {}'.format(proc_name, str(parallel_2D_task)))
            parallel_2D_task.work(task_func, task_argv)
            logger.debug('Process {} finished {}'.format(proc_name, str(parallel_2D_task)))

    # no more tasks in queue
    return True


if __name__ == '__main__':

    import time
    import sys

    log_level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(levelname)6s | %(message)s',
                        level=log_level, datefmt='%d-%b-%y %H:%M:%S')

    size = 20
    task_argv = np.array(range(0, size))

    # function to parallelize: for each (i, j) in 20x20, add the i and j position together
    def add_i_j(i, j, task_argv):

        result = task_argv[i] + task_argv[j]
        time.sleep(0.1)
        return result

    # run with supplied number of processors
    num_proc = int(sys.argv[1])
    parallel_queue = Parallel2DQueue(num_proc)
    shape_2D = (size, size)
    result2D = parallel_queue.execute_func_on_2D(add_i_j, task_argv, shape_2D)

    print(result2D.shape)
    print(result2D)
