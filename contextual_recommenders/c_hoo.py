
import numpy as np

from contextual_recommenders.contextual_recommender_templates import OnlineContextualArticleRecommender


def calculate_params(horizon, dim):
    v1, rho = 2*np.sqrt(dim), 2**(-1/dim) # these are given in the paper
    max_depth = int(np.ceil(np.log(v1*(horizon**0.5)) / np.log(1/rho))) # also given in paper, in section related to truncated tree
    return v1, rho, max_depth


def calculate_child_centers(center, dim_radii, dim_to_div):
    # calculates the child centers and new radii for each children(the same radii for both children since they are created by dividing the same dimension)
    delta, new_dim_radii, dim = np.copy(dim_radii)/2, np.copy(dim_radii), center.shape[0]
    delta[np.arange(dim) != dim_to_div] = 0
    new_dim_radii[dim_to_div] /= 2
    left_center, right_center = center - delta, center + delta
    return left_center, right_center, new_dim_radii


def contains_context(context, center, dim_radii):
    # this checks whether the given context vector is in the partition defined by the center and the radii of each dimension
    up, low = center + dim_radii, center - dim_radii
    return np.all([np.all(context <= up), np.all(context >= low)])


class ContextualXArmedBanditEfficient(OnlineContextualArticleRecommender):
    # TODO: not completed, need to discretize the arms
    def __init__(self, article_dict, dx, horizon, conf_scale=1.0):
        super().__init__(article_dict)
        self.arm_size = len(article_dict)
        self.article_ids_arr = np.array(list(self.article_dict.keys()))
        # input parameters
        self.dx, self.horizon, self.conf_scale = dx, horizon, conf_scale
        # calculate v1 and who st assumption in paper is satisfied
        self.v1, self.rho, self.max_depth = calculate_params(horizon, dx)

        # keep the tree as a dictionary of nodes: keys are (depth, index) tuples, values are linear indices
        self.tree_nodes_dict = {(0, 1): 0}
        # for each subset of the partition we will keep the geometric center, as well as the radius in every dimension
        # initialize
        self.centers, self.dim_radiis = [0.5*np.ones(shape=(dx))], [0.5*np.ones(shape=(dx))]
        # initial values for root note, all lists will expand as tree grows
        self.Bs, self.Ns, self.mus, self.Us = [np.inf], [0], [0.0], [np.inf]

        # these are used in update, after receiving reward
        self.last_played_hi_pair = (-1, -1)
        self.last_path = []

    def recommend_article(self, available_article_ids, user_features):
        """ this method determines an arm given context, follows the pseudo-code in the paper, the difference is that
        in every step one has to consider which child contains the context"""
        context = np.array(user_features)
        # we will start with the root node everytime
        hi_pair = (0, 1)
        path = [hi_pair]
        left_center, right_center, children_dim_radii, is_sel_hi_left = None, None, None, None
        while hi_pair in self.tree_nodes_dict:
            h, i, lin_ind = hi_pair[0], hi_pair[1], self.tree_nodes_dict[hi_pair] # get depth, index and linear index of the node
            left_hi, right_hi = (h + 1, 2*i - 1), (h + 1, 2*i) # get the left and right child indices (even if they are not in the tree yet)
            left_b, right_b = np.inf, np.inf
            # ucbs are infinite if they are not in the tree
            # if they are in the tree access their B values stored in the corresponding array
            if left_hi in self.tree_nodes_dict:
                left_b = self.Bs[self.tree_nodes_dict[left_hi]]
            if right_hi in self.tree_nodes_dict:
                right_b = self.Bs[self.tree_nodes_dict[right_hi]]

            # center and dimension radii of the region corresponding to this node
            center, dim_radii = self.centers[lin_ind], self.dim_radiis[lin_ind]
            # we divide partitions each time by the longest side(if multiple sides are longest, select arbitrarily)
            dim_to_divide_next = h % self.dx
            # we select the first instance of the longest side,
            # and at each level since we select it in order, which dimension to be divided is
            # directly given by the depth of the node and the num of dimensions

            # get child regions
            left_center, right_center, children_dim_radii = calculate_child_centers(center, dim_radii, dim_to_divide_next)

            # check if children contain the context
            c_in_left = contains_context(context, left_center, children_dim_radii)
            c_in_right = contains_context(context, right_center, children_dim_radii)

            is_sel_hi_left = True # this is flag, if left is chosen: True, else: False
            if c_in_left and c_in_right: # if context is in both  look at the b values to choose
                if left_b >= right_b:
                    hi_pair = left_hi
                elif left_b < right_b:
                    hi_pair, is_sel_hi_left = right_hi, False
                else:
                    z = np.random.randint(0, 2)
                    hi_pair, is_sel_hi_left = (h + 1, 2 * i - z), bool(z)
            elif c_in_left: # if only one of them contains the context then choose the one that contains
                hi_pair = left_hi
            elif c_in_right:
                hi_pair, is_sel_hi_left = right_hi, False
            else: # if no of the children contain the context, there must be some error, at least one child must contain the context
                raise ValueError('Something is wrong, context must be in at least one of the children!')

            path.append(hi_pair) # add selected child to path

            if len(path) > self.max_depth: # stop traversing if max depth is reached
                break

        sel_hi = hi_pair
        sel_center = right_center
        if sel_hi not in self.tree_nodes_dict: # if the selected node is not in tree(this does not happen when we stop due to truncation)
            self.tree_nodes_dict[sel_hi] = len(self.tree_nodes_dict) # add node to the dictionary
            self.dim_radiis.append(children_dim_radii) # add the radii info to the related array
            if is_sel_hi_left:
                sel_center = left_center
            self.centers.append(sel_center) # add the selected child's center(after determining if it is the right of the left child)
            self.Bs.append(np.inf) # new B values are inf
            self.Ns.append(0) # initialize the statistics for the node
            self.mus.append(0)
            self.Us.append(np.inf)

        up_bound_hi, low_bound_hi = sel_center + children_dim_radii, sel_center - children_dim_radii # get the upper and lower bounderies of the region selected
        arm = np.random.uniform(low=up_bound_hi[self.dx:], high=low_bound_hi[self.dx:]) # select an arm uniformly random in the region
        self.last_played_hi_pair = sel_hi # update these, we will use these when updating after receiving reward
        self.last_path = path
        return arm

    def update_statistics(self, reward):
        """ update the tree statistic according to the paper, since this is a truncated version the update is much simpler see the related section in
        paper """
        for hi_pair in self.last_path: # for each node in the previously followed path
            h, i, lin_ind = hi_pair[0], hi_pair[1], self.tree_nodes_dict[hi_pair]
            self.Ns[lin_ind] += 1 # increase # of times played
            self.mus[lin_ind] = float((1 - (1/self.Ns[lin_ind])) * self.mus[lin_ind] + (reward / self.Ns[lin_ind])) # update mean rew est for the region
            conf_terms = np.sqrt(2 * np.log(self.horizon) / self.Ns[lin_ind]) + self.v1 * (self.rho ** h) # update U terms
            self.Us[lin_ind] = self.mus[lin_ind] + self.conf_scale * conf_terms

        for hi_pair in reversed(self.last_path): # now starting from the leaf traverse path upwards
            h, i, lin_ind = hi_pair[0], hi_pair[1], self.tree_nodes_dict[hi_pair]
            left_hi, right_hi = (h + 1, 2 * i - 1), (h + 1, 2 * i)
            is_h_leaf = (left_hi not in self.tree_nodes_dict) and (right_hi not in self.tree_nodes_dict)
            if is_h_leaf: # if the node is a leaf just update B value as follows
                self.Bs[lin_ind] = self.Us[lin_ind]
            else: # if not then use the B values of the children to update the B value, as well as the U value
                left_b, right_b = np.inf, np.inf
                if left_hi in self.tree_nodes_dict:
                    left_b = self.Bs[self.tree_nodes_dict[left_hi]]
                if right_hi in self.tree_nodes_dict:
                    right_b = self.Bs[self.tree_nodes_dict[right_hi]]
                self.Bs[lin_ind] = np.min([self.Us[lin_ind], np.max([left_b, right_b])])