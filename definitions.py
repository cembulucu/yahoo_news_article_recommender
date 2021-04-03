import os

DEFINITIONS_PATH = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(DEFINITIONS_PATH)
OG_DATA_DIR = os.path.join(ROOT_DIR, 'data')
OG_DATA_FILE_PATHS = [os.path.join(OG_DATA_DIR, f_str) for f_str in os.listdir(OG_DATA_DIR)]
RESTRUCTURED_DATA_FILE_PATH = os.path.join(ROOT_DIR, 'restructured_data', 'restructured_clicks.dat')
ARTICLE_DICT_JSON_PATH = os.path.join(ROOT_DIR, 'restructured_data', 'article_dict.json')

if __name__ == '__main__':
    print(DEFINITIONS_PATH)
    print(ROOT_DIR)
    print(OG_DATA_DIR)
    for d in OG_DATA_FILE_PATHS:
        print(d)
    print(RESTRUCTURED_DATA_FILE_PATH)
    print(ARTICLE_DICT_JSON_PATH)