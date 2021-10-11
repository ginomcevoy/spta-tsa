import importlib
import os
import sys


def getClass(full_class_name):
    '''
    Given a full class name (package.module.class), retrieve the class.
    '''
    # To do this, we need the module and the class name
    # The 'class' parameter has the full name, so we split it
    # example: spta.dataset.DatasetCSFR -> spta.dataset and DatasetCSFR
    name_parts = full_class_name.split('.')
    class_name = name_parts[-1]
    module_name = '.'.join(name_parts[0:-1])

    # the class is an attribute of its containing module
    a_module = importlib.import_module(module_name)
    a_class = getattr(a_module, class_name)
    return a_class


def classExists(full_class_name):
    '''
    Returns True iff the the class_name can be found.
    '''
    # just try to get the class object
    exists = True
    try:
        getClass(full_class_name)
    except Exception:
        exists = False

    return exists


def createInstanceWithArgs(full_class_name, **kwargs):
    '''
    Given a full class name (package.module.class) and keyword arguments,
    create an instance of the class by passing the args to the constructor.
    '''
    a_class = getClass(full_class_name)
    instance = a_class(**kwargs)
    return instance


def getModuleDirectory(module_name):
    '''
    Given a module name as string, return the directory where the module is located.
    '''
    # if module not available, try to import it manually
    if module_name not in sys.modules:
        importlib.import_module(module_name)

    actual_module = sys.modules[module_name]
    return os.path.dirname(actual_module.__file__)
