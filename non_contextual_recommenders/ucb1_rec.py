from non_contextual_recommenders.non_contextual_recommender_templates import OnlineNonContextualArticleRecommender
import numpy as np


class UCB1Recommender(OnlineNonContextualArticleRecommender):
    # TODO: add UCB1_tuned option
    def __init__(self, article_dict, conf_scale=1.0):
        super().__init__(article_dict)
        self.conf_scale = conf_scale
        self.article_ids_arr = np.array(list(self.article_dict.keys()))
        self.sums = np.zeros(shape=(len(article_dict),))
        self.counts = np.zeros(shape=(len(article_dict),))
        self.counts_all = 0
        self.last_played_arm_ind = None

    def recommend_article(self, available_article_ids):
        with np.errstate(divide='ignore', invalid='ignore'):
            means = self.sums/self.counts
            confs = np.sqrt((2*np.log(self.counts_all)) / self.counts)
        ucbs = means + self.conf_scale*confs

        ucbs[np.isnan(ucbs)] = np.inf
        ucbs[~np.isin(self.article_ids_arr, available_article_ids)] = -np.inf

        # print('UCB1 ucb: ', np.max(ucbs))
        max_ucb_inds = np.atleast_1d(np.squeeze(np.argwhere(ucbs == np.max(ucbs))))
        random_max_ind = np.random.choice(max_ucb_inds)
        rec_art_id = self.article_ids_arr[random_max_ind]
        self.last_played_arm_ind = random_max_ind
        return rec_art_id

    def update_statistics(self, reward):
        self.sums[self.last_played_arm_ind] += reward
        self.counts[self.last_played_arm_ind] += 1
        self.counts_all += 1