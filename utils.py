from time import strftime, gmtime
import logging, os, random
import torch


def NEFTune(model, noise_alpha=5):
    def noised_embed(orig_embed, noise_alpha):
        def new_func(x):
            # during training, we add noise to the embedding
            # during generation, we don't add noise to the embedding
            if model.training:
                embed_init = orig_embed(x)
                dims = torch.tensor(embed_init.size(1) * embed_init.size(2))
                mag_norm = noise_alpha / torch.sqrt(dims)
                return embed_init + torch.zeros_like(embed_init).uniform_(-mag_norm, mag_norm)
            else:
                return orig_embed(x)

        return new_func

    ##### NOTE: this is for a BERT model #####
    ##### For a different model, you need to change the attribute path to the embedding #####
    ##### 递归定义有问题
    model.base_model.model.bert.embeddings.word_embeddings.forward = noised_embed(
        model.base_model.model.bert.embeddings.word_embeddings, noise_alpha)

    return model

def neftune_post_forward_hook(module, input, output):
    """
    Implements the NEFTune forward pass for the model using forward hooks. Note this works only for
    torch.nn.Embedding layers. This method is slightly adapted from the original source code
    that can be found here: https://github.com/neelsjain/NEFTune

    Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```

    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set
            `module.neftune_noise_alpha` to the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    if module.training:
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
        output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    return output

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
        train_filelist.write(fl + "\n")
    for fl in filelist[1024:1024 + 128]:
        val_filelist.write(fl + "\n")


if __name__ == "__main__":
    filelistcol("D:/pythonwork/W2NER/data/OriginalFiles/data_origin")
