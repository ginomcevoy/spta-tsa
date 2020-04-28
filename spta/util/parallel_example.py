# https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
import numpy as np
import time
from multiprocessing import Pool, RawArray, current_process

# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(X, X_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape

def worker_func(i):
    # Simply computes the sum of the i-th row of the input matrix X
    print('Working {} -> {}'.format(current_process().name, i))
    X_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    time.sleep(1) # Some heavy computations
    return np.asscalar(np.sum(X_np[i,:]))

# We need this check for Windows to prevent infinitely spawning new child
# processes.
if __name__ == '__main__':
    X_shape = (16, 1000000)
    # Randomly generate some data
    data = np.random.randn(*X_shape)
    X = RawArray('d', X_shape[0] * X_shape[1])
    # Wrap X as an numpy array so we can easily manipulates its data.
    X_np = np.frombuffer(X).reshape(X_shape)
    # Copy data to our shared array.
    np.copyto(X_np, data)
    # Start the process pool and do the computation.
    # Here we pass X and X_shape to the initializer of each worker.
    # (Because X_shape is not a shared variable, it will be copied to each
    # child process.)
    with Pool(processes=4, initializer=init_worker, initargs=(X, X_shape)) as pool:
        result = pool.map(worker_func, range(X_shape[0]))
        print('Results (pool):\n', np.array(result))
    # Should print the same results.
    print('Results (numpy):\n', np.sum(X_np, 1))
