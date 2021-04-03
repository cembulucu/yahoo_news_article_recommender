import numpy as np
import sklearn.metrics as sk_metrics
import scipy.special as sp_special
import itertools

from contextual_recommenders.contextual_recommender_templates import OnlineContextualArticleRecommender


def calculate_set_w_st_vcw(v_dx_x, v_2dx_x):
    w_inds_list, w_tuples_list = [], []
    for v_i, v in enumerate(v_dx_x):
        # for each dx_bar tuple keep a list of 2dx_bar tuples that are supersets of the dx_bar tuple(for both tuple index and value)
        w_inds_list_v, w_tuples_list_v = [], []
        for w_i, w in enumerate(v_2dx_x):
            is_subset = set(w).issuperset(set(v))
            if is_subset:
                w_inds_list_v.append(w_i)
                w_tuples_list_v.append(w)
        w_inds_list.append(w_inds_list_v)
        w_tuples_list.append(w_tuples_list_v)
    return np.array(w_inds_list), np.array(w_tuples_list)


def calculate_partial_partitions(m, d, tuples):
    d_bar = tuples.shape[1] # infer d_bar from tuple shape
    centers_vector = np.arange(m) / m + (0.5 / m) # contains a partition for a single dimension
    central_val_perms = np.array(list(itertools.product(centers_vector, repeat=d_bar))) # permutate for num of relevant dimensions
    part_set = np.zeros(shape=(0, d)) # initialize partition set
    for v in tuples:
        # for each tuple create whole set of point, each with value 0.5
        y = 0.5 * np.ones(shape=(central_val_perms.shape[0], d))
        # set the "assumed" relevant dimensions s.t. they are partitioned
        y[:, v] = central_val_perms
        part_set = np.append(part_set, y, axis=0)
    return part_set


class CMABRLwDiscreteArmsRecommender(OnlineContextualArticleRecommender):
    def __init__(self, article_dict, horizon, dx, dx_bar, lip_c, conf_scale):
        super().__init__(article_dict)
        self.arm_size = len(article_dict)
        self.article_ids_arr = np.array(list(self.article_dict.keys()))
        self.horizon, self.dx, self.dx_bar, self.lip_c, self.conf_scale = horizon, dx, dx_bar, lip_c, conf_scale
        if horizon == 100000:  # this if statement is needed since 100000**(1 / 5) returns 10.000000000000002
            self.m = 10
        else:
            self.m = int(np.ceil((horizon/self.arm_size) ** (1 / (2 + 2 * dx_bar))))

        # calculate sets of tuples: num of tuples are (d choose d_bar) for each
        self.V_dxb_x = np.array(list(itertools.combinations(np.arange(dx), dx_bar)))
        self.V_2dxb_x = np.array(list(itertools.combinations(np.arange(dx), 2 * dx_bar)))
        self.W_centers = calculate_partial_partitions(self.m, self.dx, self.V_2dxb_x)

        # calculate which 2dx_bar tuples are supersets of which dx_bar tuples
        self.vCw_inds, self.vCw_tuples = calculate_set_w_st_vcw(self.V_dxb_x, self.V_2dxb_x)

        # num of comparisons needed for each v (w vs w_prime comparisons needed in calculating relevance set)
        self.full_abs_diff_count_per_v = self.vCw_inds.shape[1] * (self.vCw_inds.shape[1] - 1) // 2

        # this for loop prebuilds all the combinations of 2dx_bar tuples that are supersets of corresponding dx_bar needed for relevance set
        # (this job is time consuming so instead of doing this every round we calculate the pairs now and iterate in a single loop)
        self.w_wp_comb_lists_list, self.w_wp_inds_comb_lists_list = [], []
        for v_i, v in enumerate(self.V_dxb_x):
            w_wp_comb_list = list(itertools.combinations(self.vCw_tuples[v_i], 2))
            w_wp_inds_comb_list = list(itertools.combinations(self.vCw_inds[v_i], 2))
            self.w_wp_comb_lists_list.append(w_wp_comb_list)
            self.w_wp_inds_comb_lists_list.append(w_wp_inds_comb_list)

        # initialize counters
        self.sample_counts = np.zeros(shape=(self.arm_size, self.W_centers.shape[0]))
        self.sample_means = np.zeros(shape=(self.arm_size, self.W_centers.shape[0]))

        # initialize some constants
        self.c_bar = sp_special.comb(dx - 1, 2 * dx_bar - 1, exact=False)
        self.u_nom = 2 + 4 * np.log(2 * self.arm_size * (self.horizon ** 1.5) * self.c_bar * (self.m ** (2 * self.dx_bar)))
        self.context_err = lip_c * np.sqrt(dx_bar) / self.m

        # keep necessary info for last played round to be able to update when reward is received(these values are just placeholders)
        self.last_played_arm_ind = -1
        self.last_observed_context_pw_inds = []

    def recommend_article(self, available_article_ids, user_features):
        context = np.array(user_features)
        pw_inds = self.find_pws_context_resides(context)
        cys = self.determine_cys(pw_inds)
        ucbs = self.calculate_ucbs(cys, pw_inds)
        ucbs[~np.isin(self.article_ids_arr, available_article_ids)] = -np.inf

        # if multiple arms have the same highest ucb, select randomly among them
        # print('cmab_rl ucb: ', np.max(ucbs))
        max_ucb_inds = np.atleast_1d(np.squeeze(np.argwhere(ucbs == np.max(ucbs))))
        random_max_ind = np.random.choice(max_ucb_inds)
        rec_art_id = self.article_ids_arr[random_max_ind]

        self.last_played_arm_ind = random_max_ind  # store the selected arm and pws that context hits for update after receiving reward
        self.last_observed_context_pw_inds = pw_inds

        return rec_art_id

    def update_statistics(self, reward):
        # get selected arm and pws that context hits from prev round
        y_ind, pw_inds = self.last_played_arm_ind, self.last_observed_context_pw_inds
        # get related counters
        y_means, y_counts = self.sample_means[y_ind, pw_inds], self.sample_counts[y_ind, pw_inds]
        # update related counters
        self.sample_means[y_ind, pw_inds] = (y_means * y_counts + reward) / (y_counts + 1)
        # updating for all ws allows for faster convergence
        self.sample_counts[y_ind, pw_inds] = y_counts + 1

    def find_pws_context_resides(self, context):
        """ this method find the pw subsets that context resides """
        # get number of 2dx_tuples and number of pws for each 2dx_tuple
        num_w, num_parts_each_w = self.V_2dxb_x.shape[0], self.m ** (2*self.dx_bar)
        # in this array we will store which pw is "active" for each w
        pw_inds = np.zeros(shape=(num_w, ), dtype=np.int)
        # calculate distance from the context vector to each pw
        distances_to_pws = np.squeeze(sk_metrics.pairwise_distances(np.expand_dims(context, axis=0), self.W_centers, metric='cityblock'))
        # in this for loop for each 2dx_tuple,
        # find the pw that context resides in(by looking at the min distance to pw centers among each w)
        for w_ind, w_tuple in enumerate(self.V_2dxb_x):
            pws_selection = np.arange(start=w_ind*num_parts_each_w, stop=(w_ind+1)*num_parts_each_w)
            dists_w = distances_to_pws[pws_selection]
            min_dist_ind_w = np.argmin(dists_w)
            pw_inds[w_ind] = w_ind*num_parts_each_w + min_dist_ind_w
        return pw_inds

    def determine_cys(self, pw_inds):
        """ this method determines the cys for each arm y, this is the core of the algorithm """
        # get counters
        sample_means_pw = self.sample_means[:, pw_inds]
        sample_counts_pw = self.sample_counts[:, pw_inds]

        # avoid divide by zero warning(nothing important, just prevents np printing warning)
        # as u_nom is never 0, we will get infinite ucbs at start. calculate confidence radii.
        with np.errstate(divide='ignore'):
            u_vals_pw = self.conf_scale * np.sqrt(self.u_nom / sample_counts_pw)
        # placeholder for cys array, we will store estimated relevant dimensions for each y in this array
        cys = -1*np.ones(shape=(self.arm_size, ), dtype=np.int)

        # for each arm, we will calculate dx_tuples that are in relevance set and store their sigmas
        for y_i, y in enumerate(self.article_ids_arr):
            sigmas, valid_vs = [], []
            # for each dx_tuple we will check every combination of 2dx_tuples
            for v_i, v in enumerate(self.V_dxb_x):
                # and if so store the abs_diff(for sigma calculation this is needed, we store it here to avoid calculating again)
                abs_diffs = []
                # get w and w_prime combinations that are supersets of v
                w_wp_comb_list = self.w_wp_comb_lists_list[v_i]
                w_wp_inds_comb_list = self.w_wp_inds_comb_lists_list[v_i]
                # for each combination, check if they are satisfying relevance relation
                for w_inds, w_wp in zip(w_wp_inds_comb_list, w_wp_comb_list):
                    w_i, wp_i, w, wp = w_inds[0], w_inds[1], w_wp[0], w_wp[1]
                    abs_diff = np.abs(sample_means_pw[y_i, w_i] - sample_means_pw[y_i, wp_i])
                    up_bound = 2 * self.context_err + u_vals_pw[y_i, w_i] + u_vals_pw[y_i, wp_i]
                    is_abs_diff_in_bounds = abs_diff <= up_bound
                    # if even one combination violates relevance relation, do not search for more
                    if not is_abs_diff_in_bounds:
                        break
                    # add only if relevance relation is satisfied
                    abs_diffs.append(abs_diff)
                # if even one combination violates relevance relation, abs_diffs will not be "full"
                if len(abs_diffs) == self.full_abs_diff_count_per_v:
                    # this happens when 2dx_bar = dx
                    if len(abs_diffs) == 0:
                        sigmas.append(0)
                    else:
                        # add to sigmas only if v is satisfying relevance relation
                        sigmas.append(np.max(abs_diffs))
                    valid_vs.append(v_i)
            # not even a single v satisfies confident event, then select cy randomly(only happens in unconfident event)
            if len(sigmas) == 0:
                cys[y_i] = np.random.randint(self.V_dxb_x.shape[0], size=1)
            else:
                # if multiple v's are satisfying the relevance relation select the one with minimum sigma
                # if multiple sigmas are minimum then select randomly among them
                winner_v_inds = np.argwhere(sigmas == np.min(sigmas))
                if winner_v_inds.shape[0] > 1:
                    winner = np.squeeze(winner_v_inds[np.random.randint(winner_v_inds.shape[0], size=1)])
                else:
                    winner = np.squeeze(winner_v_inds)
                cys[y_i] = valid_vs[int(winner)]
        return cys

    def calculate_ucbs(self, cys, pw_inds):
        """ this method calculates the final ucbs for each arm """
        # get counters
        sample_means_pw = self.sample_means[:, pw_inds]
        sample_counts_pw = self.sample_counts[:, pw_inds]
        # calculate confidence radii(conf_scale is done late this time, not that it matters)
        with np.errstate(divide='ignore'):
            u_vals_pw = np.sqrt(self.u_nom / sample_counts_pw)
        # get max for each w
        u_vals_pw_ys = np.max(u_vals_pw, axis=-1)
        # placeholder for estimated means
        est_mean_ys = -1 * np.ones(shape=(self.arm_size, ))
        # for each arm get only the 2dx_bar tuples that contain the estimated relevant context dx_bar tuple
        v_2dx_x_superset_v = self.vCw_inds[cys]
        # this for loop calculates the estimated means for each arm
        for y_i, w_list in enumerate(v_2dx_x_superset_v):
            # get sample means and counts from pws of 2dx_tuples
            sample_means_pw_y = sample_means_pw[y_i, w_list]
            sample_counts_pw_y = sample_counts_pw[y_i, w_list]
            # calculate estimated mean
            with np.errstate(invalid='ignore'):
                est_mean_y = np.sum(sample_means_pw_y*sample_counts_pw_y) / np.sum(sample_counts_pw_y)
            # nan would mean 0/0 due to sample_counts_pw_y being all zeros
            # in this case the confidence radii is inf anyways so just set the mean to 0
            if np.isnan(est_mean_y):
                est_mean_ys[y_i] = 0
            else:
                est_mean_ys[y_i] = est_mean_y

        ucbs = est_mean_ys + self.conf_scale * 5 * u_vals_pw_ys  # calculate final scaled ucb
        return ucbs