'''
Data loading, cleaning, preprocessing, and chunking.
Any text normalization or embedding preparation steps.
'''
'''
Here, we have just one data source - a text file on cat facts. And each line in 
the file is a chunk. There are no further embedding preparation steps.
'''
from db import *

def pipeline_get_raw_data():
    print("Loading raw data...")
    with open("cat-facts.txt", "r") as file:
        facts = file.readlines()
    return facts


