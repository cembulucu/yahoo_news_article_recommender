
from definitions import ARTICLE_DICT_JSON_PATH
from non_contextual_recommenders.non_contextual_recommender_templates import OnlineNonContextualArticleRecommender
import numpy as np
import json


class RandomArticleSelector(OnlineNonContextualArticleRecommender):
    # TODO: what to do about the errored article id
    def recommend_article(self, available_article_ids):
        return np.random.choice(available_article_ids)
