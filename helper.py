import os
import pickle
from file_utils import FileUtils
import re


def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.lower()


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    print('loading data set')
    text = load_data(dataset_path)

    print('creating token punctuation dict')
    token_dict = token_lookup()

    print('replacing tokens')
    text = multiple_replace(token_dict, text)

    print('spliting text')
    text = text.split()

    print('creating lookup tables')
    vocab_to_int, int_to_vocab = create_lookup_tables(text)

    print('saving data.')
    pickle_dump(([vocab_to_int[word] for word in text], vocab_to_int, int_to_vocab, token_dict), 'preprocess.p')


def multiple_replace(dict, text): 

  """ Replace in 'text' all occurences of any key in the given
  dictionary by its corresponding value.  Returns the new tring.""" 

  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle_load('preprocess.p')


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))

def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, FileUtils(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(FileUtils(f))