from contextual_recommenders.contextual_recommender_templates import OnlineContextualArticleRecommender
import numpy as np
import itertools
import sklearn.metrics as skmetrics


class ContextualUCB1Recommender(OnlineContextualArticleRecommender):
    def __init__(self, article_dict, horizon, context_dim, conf_scale=1.0):
        super().__init__(article_dict)
        self.num_arms = len(article_dict)
        self.article_ids_arr = np.array(list(self.article_dict.keys()))
        self.horizon, self.conf_scale = horizon, conf_scale
        self.m = np.ceil((horizon/self.num_arms) ** (1 / (2 + context_dim))).astype(int)
        centers_vector = np.arange(self.m) / self.m + (0.5 / self.m)  # calculate centers for one dimension
        self.context_centers = np.array(list(itertools.product(centers_vector, repeat=context_dim)))  # extend centers for all dimensions
        self.num_context_subsets = self.context_centers.shape[0]

        self.sample_counts = np.zeros(shape=(self.num_context_subsets, self.num_arms))
        self.sample_means = np.zeros(shape=(self.num_context_subsets, self.num_arms))

        self.last_arrived_context_ind, self.last_played_arm_ind = -1, -1
        self.all_plays_count = 0

    def recommend_article(self, available_article_ids, user_features):
        context = np.array(user_features)
        diss_to_centers = np.squeeze(skmetrics.pairwise_distances(np.expand_dims(context, axis=0), self.context_centers))
        min_dist_ind = np.argmin(diss_to_centers)

        rel_sample_means, rel_sample_counts = self.sample_means[min_dist_ind], self.sample_counts[min_dist_ind]

        # calculate confidence terms (UCB1)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_confs = np.sqrt(2 * np.log(self.all_plays_count) / rel_sample_counts)

        rel_ucbs = rel_sample_means + self.conf_scale * rel_confs
        rel_ucbs[np.isnan(rel_ucbs)] = np.inf
        rel_ucbs[~np.isin(self.article_ids_arr, available_article_ids)] = -np.inf

        # print('contextual ucb1 ucb: ', np.max(rel_ucbs))
        max_ucb_inds = np.atleast_1d(np.squeeze(np.argwhere(rel_ucbs == np.max(rel_ucbs))))
        random_max_ind = np.random.choice(max_ucb_inds)
        rec_art_id = self.article_ids_arr[random_max_ind]

        self.last_played_arm_ind = random_max_ind
        self.last_arrived_context_ind = min_dist_ind

        return rec_art_id


    def update_statistics(self, reward):
        # update counters for the last played arm and context
        x, y = self.last_arrived_context_ind, self.last_played_arm_ind
        self.sample_means[x, y] = (self.sample_means[x, y] * self.sample_counts[x, y] + reward) / (self.sample_counts[x, y] + 1)
        self.sample_counts[x, y] = self.sample_counts[x, y] + 1
        self.all_plays_count += 1


class ContextualIUPRecommender(OnlineContextualArticleRecommender):
    def __init__(self, article_dict, horizon, context_dim, conf_scale=1.0):
        super().__init__(article_dict)
        self.num_arms = len(article_dict)
        self.article_ids_arr = np.array(list(self.article_dict.keys()))
        self.horizon, self.conf_scale = horizon, conf_scale
        self.m = np.ceil((horizon/self.num_arms) ** (1 / (2 + context_dim))).astype(int)
        centers_vector = np.arange(self.m) / self.m + (0.5 / self.m)  # calculate centers for one dimension
        self.context_centers = np.array(list(itertools.product(centers_vector, repeat=context_dim)))  # extend centers for all dimensions
        self.num_context_subsets = self.context_centers.shape[0]

        self.sample_counts = np.zeros(shape=(self.num_context_subsets, self.num_arms))
        self.sample_means = np.zeros(shape=(self.num_context_subsets, self.num_arms))

        self.last_arrived_context_ind, self.last_played_arm_ind = -1, -1
        self.all_plays_count = 0

    def recommend_article(self, available_article_ids, user_features):
        context = np.array(user_features)
        diss_to_centers = np.squeeze(skmetrics.pairwise_distances(np.expand_dims(context, axis=0), self.context_centers))
        min_dist_ind = np.argmin(diss_to_centers)

        rel_sample_means, rel_sample_counts = self.sample_means[min_dist_ind], self.sample_counts[min_dist_ind]

        # calculate confidence terms (IUP)
        with np.errstate(divide='ignore', invalid='ignore'):
            inside_sqrt_1 = 2 / rel_sample_counts
            inside_sqrt_2 = 1 + 2 * np.log(2 * self.num_arms * self.num_context_subsets * (self.horizon ** 1.5))
            rel_confs = np.sqrt(inside_sqrt_1 * inside_sqrt_2)

        rel_ucbs = rel_sample_means + self.conf_scale * rel_confs
        rel_ucbs[np.isnan(rel_ucbs)] = np.inf
        rel_ucbs[~np.isin(self.article_ids_arr, available_article_ids)] = -np.inf

        # print('contextual iup ucb: ', np.max(rel_ucbs))
        max_ucb_inds = np.atleast_1d(np.squeeze(np.argwhere(rel_ucbs == np.max(rel_ucbs))))
        random_max_ind = np.random.choice(max_ucb_inds)
        rec_art_id = self.article_ids_arr[random_max_ind]

        self.last_played_arm_ind = random_max_ind
        self.last_arrived_context_ind = min_dist_ind

        return rec_art_id


    def update_statistics(self, reward):
        # update counters for the last played arm and context
        x, y = self.last_arrived_context_ind, self.last_played_arm_ind
        self.sample_means[x, y] = (self.sample_means[x, y] * self.sample_counts[x, y] + reward) / (self.sample_counts[x, y] + 1)
        self.sample_counts[x, y] = self.sample_counts[x, y] + 1
        self.all_plays_count += 1