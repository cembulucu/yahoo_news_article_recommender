from non_contextual_recommenders.non_contextual_recommender_templates import OnlineNonContextualArticleRecommender
import numpy as np


class MaxMeanArticleSelector(OnlineNonContextualArticleRecommender):
    def __init__(self, article_dict):
        super().__init__(article_dict)
        self.article_ids_arr = np.array(list(self.article_dict.keys()))
        self.sums = np.zeros(shape=(len(article_dict),))
        self.counts = np.zeros(shape=(len(article_dict),))
        self.last_played_arm_ind = None

    def recommend_article(self, available_article_ids):
        with np.errstate(divide='ignore', invalid='ignore'):
            means = self.sums/self.counts
        means[np.isnan(means)] = np.inf
        means[~np.isin(self.article_ids_arr, available_article_ids)] = -np.inf
        max_mean_inds = np.atleast_1d(np.squeeze(np.argwhere(means == np.max(means))))
        random_max_ind = np.random.choice(max_mean_inds)
        rec_art_id = self.article_ids_arr[random_max_ind]
        self.last_played_arm_ind = random_max_ind
        return rec_art_id

    def update_statistics(self, reward):
        self.sums[self.last_played_arm_ind] += reward
        self.counts[self.last_played_arm_ind] += 1
