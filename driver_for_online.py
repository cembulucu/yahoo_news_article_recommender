import time

from contextual_recommenders.cmab_rl import CMABRLwDiscreteArmsRecommender
from contextual_recommenders.simple_UCB_based_contextual import ContextualUCB1Recommender, ContextualIUPRecommender
from definitions import RESTRUCTURED_DATA_FILE_PATH, ARTICLE_DICT_JSON_PATH
from non_contextual_recommenders.max_mean_selector import MaxMeanArticleSelector
from non_contextual_recommenders.random_article_selector import RandomArticleSelector
from non_contextual_recommenders.ucb1_rec import UCB1Recommender
from recommendation_env import ArticleRecommendationEnvironment
import json
import numpy as np
from utilities import replace_string_keys_with_int_keys


def main():
    horizon, verbose_period = int(1e7), int(1e3)
    alg_horizon = horizon//20
    print('Horizon: ', horizon, ', algorithm horizon: ', alg_horizon, 'verbose period: ', verbose_period)
    rec_env = ArticleRecommendationEnvironment(data_file_path=RESTRUCTURED_DATA_FILE_PATH)

    # rec_env = ArticleRecommendationEnvironment(data_file_path='demo_res_dat.dat')

    with open(ARTICLE_DICT_JSON_PATH, 'r') as f:
    # with open('demo_art_dict.json', 'r') as f:
        article_dict = replace_string_keys_with_int_keys(json.load(f))

    # non_contextual_algs = [RandomArticleSelector(article_dict=article_dict),
    #                        MaxMeanArticleSelector(article_dict=article_dict),
    #                        UCB1Recommender(article_dict=article_dict, conf_scale=1)]
    non_contextual_algs = []
    non_contextual_alg_rews = np.zeros(shape=(len(non_contextual_algs)))
    non_contextual_alg_counts = np.zeros(shape=(len(non_contextual_algs)))

    # contextual_algs = [ContextualUCB1Recommender(article_dict, horizon=alg_horizon, context_dim=5, conf_scale=1),
    #                    ContextualIUPRecommender(article_dict, horizon=alg_horizon, context_dim=5, conf_scale=1),
    #                    CMABRLwDiscreteArmsRecommender(article_dict, horizon=alg_horizon, dx=5, dx_bar=2, lip_c=1.0, conf_scale=1)]
    contextual_algs = [CMABRLwDiscreteArmsRecommender(article_dict, horizon=alg_horizon, dx=5, dx_bar=2, lip_c=1.0, conf_scale=1)]
    contextual_alg_rews = np.zeros(shape=(len(contextual_algs)))
    contextual_alg_counts = np.zeros(shape=(len(contextual_algs)))

    start_time = time.time()
    for t in range(horizon):
        _, u_feats, aa_ids = rec_env.get_next_instance()
        if u_feats is None:
            break  # end of samples is reached

        for i, non_cont_alg in enumerate(non_contextual_algs):
            rec_id = non_cont_alg.recommend_article(available_article_ids=aa_ids)
            rew = rec_env.get_reward(rec_id)
            if rew is not None:
                non_cont_alg.update_statistics(rew)
                non_contextual_alg_rews[i] += rew
                non_contextual_alg_counts[i] += 1

        for i, cont_alg in enumerate(contextual_algs):
            rec_id = cont_alg.recommend_article(available_article_ids=aa_ids, user_features=u_feats)
            rew = rec_env.get_reward(rec_id)
            if rew is not None:
                cont_alg.update_statistics(rew)
                contextual_alg_rews[i] += rew
                contextual_alg_counts[i] += 1

        if t % verbose_period == 0:
            now_time = time.time()
            time_diff = now_time - start_time
            print('\nInstances that are considered so far: ', t + 1, ', time it took: {} seconds'.format(time_diff))
            with np.errstate(divide='ignore', invalid='ignore'), np.printoptions(precision=2, suppress=True):
                for i, non_cont_alg in enumerate(non_contextual_algs):
                    mean_rew = non_contextual_alg_rews[i]/non_contextual_alg_counts[i]
                    print('class: ', non_cont_alg.__class__.__name__,
                          ',\tnum samples = ', non_contextual_alg_counts[i],
                          ',\ttotal rew = ', non_contextual_alg_rews[i],
                          ',\tmean rew = ', mean_rew)

                for i, cont_alg in enumerate(contextual_algs):
                    mean_rew = contextual_alg_rews[i] / contextual_alg_counts[i]
                    print('class: ', cont_alg.__class__.__name__,
                          ',\tnum samples = ', contextual_alg_counts[i],
                          ',\ttotal rew = ', contextual_alg_rews[i],
                          ',\tmean rew = ', mean_rew)





if __name__ == '__main__':
    main()