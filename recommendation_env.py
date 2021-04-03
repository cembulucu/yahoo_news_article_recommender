import json

from definitions import RESTRUCTURED_DATA_FILE_PATH
import numpy as np


class ArticleRecommendationEnvironment:
    def __init__(self, data_file_path):
        self.dat_file = open(data_file_path, 'r')

        self.stream_gen_obj = self.stream_generator()
        self.ts, self.sa_id, self.click, self.u_feats, self.aa_ids = None, None, None, None, None

    def get_next_instance(self):
        line_ind, line_json_str = next(self.stream_gen_obj, (None, None)) # set default value to None, yield None when end is reached
        if line_ind is None:
            return None, None, None # end is reached, return None
        line_json_obj = json.loads(line_json_str)

        self.sa_id = line_json_obj['shown_article_id']
        self.click = line_json_obj['click']
        u_feats = np.array(line_json_obj['user_features'])
        aa_ids = np.array(line_json_obj['available_article_ids'])

        return line_ind, u_feats, aa_ids

    def get_reward(self, selected_article_id):
        reward = self.click if selected_article_id == self.sa_id else None
        return reward

    def stream_generator(self):
        for i, line in enumerate(self.dat_file):
            yield i, line
