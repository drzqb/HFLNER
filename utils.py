from time import strftime, gmtime
import logging, os, random


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


def filelistcol(originaldir):
    filelist = []

    for dirname in os.listdir(originaldir):
        first_path = os.path.join(originaldir, dirname)

        for filename in os.listdir(first_path):
            if "txtoriginal" in filename:
                file_path = os.path.join(first_path, filename)
                with open(file_path, "r", encoding="utf-8") as fr:
                    text_title = fr.readline().rstrip()
                    lt = len(text_title)
                    if lt > 0:
                        filelist.append(file_path)

    random.shuffle(filelist)

    train_filelist = open("data/train_filelist.txt", mode="w", encoding="utf-8")
    val_filelist = open("data/val_filelist.txt", mode="w", encoding="utf-8")

    for fl in filelist[:1024]:
        train_filelist.write(fl+"\n")
    for fl in filelist[1024:1024 + 128]:
        val_filelist.write(fl+"\n")


if __name__ == "__main__":
    filelistcol("D:/pythonwork/W2NER/data/OriginalFiles/data_origin")
