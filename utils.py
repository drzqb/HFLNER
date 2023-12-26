from time import strftime, gmtime
import logging

def format_time(time):
    if time >= 3600:
        return strftime("%H:%M:%S", gmtime(time))
    else:
        return strftime("%M:%S", gmtime(time))


def create_logger(name, filename):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename=filename, mode="a", encoding="utf-8")

    simple_formatter = logging.Formatter(fmt="%(asctime)s %(message)s",
                                         datefmt="%H:%M:%S",
                                         )
    complex_formatter = logging.Formatter(fmt="%(asctime)s %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S",
                                          )

    consoleHandler.setFormatter(simple_formatter)
    fileHandler.setFormatter(complex_formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger