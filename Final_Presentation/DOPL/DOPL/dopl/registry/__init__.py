TRAINING_FUNCTIONS = {}


def register_training_function(name):
    def wrapper(func):
        TRAINING_FUNCTIONS[name] = func
        return func

    return wrapper
