import math
from definitions import OG_DATA_FILE_PATHS, RESTRUCTURED_DATA_FILE_PATH, ARTICLE_DICT_JSON_PATH
import json


# parse a features str and return extracted features
def parse_features(feats_segment):
    feat_seg_split = feats_segment.strip().split(' ')
    id = int(feat_seg_split[0]) if feat_seg_split[0].isdigit() else feat_seg_split[0]  # id is 'user' for user and an int for an article
    feats = [float(feat_str.split(':')[-1]) for feat_str in feat_seg_split[1:]]  # remove feature index chars and store as float
    feats = feats[0:-1]  # remove biasses, we do not need them
    return id, feats


# parse a line and return timestamps, shown article id, click info, user features, and available articles
def parse_line(line):
    aa_dict = {}

    split_by_vert_line = line.split('|')

    # parse header info
    head_split = split_by_vert_line[0].split(' ')
    ts = int(head_split[0])  # timestamp
    sa_id = int(head_split[1])  # shown article id
    click = int(head_split[2])  # click or not

    # parse user info, user ids are anonymized hence unnecessary, only get user features
    _, u_feats = parse_features(feats_segment=split_by_vert_line[1])

    # parse available articles info and add to dict
    for aa_feats_str in split_by_vert_line[2:]:
        aa_id, feats = parse_features(feats_segment=aa_feats_str)
        aa_dict[aa_id] = feats

    return ts, sa_id, click, u_feats, aa_dict


# merges data into single file while storing features of articles in a dict, which is more memory efficient
def merge_data_into_single_file(data_file_paths, merge_file_path, article_dict_json_path, verbose_period=math.inf):
    article_dict = {}

    with open(file=merge_file_path, mode='a') as merge_file_obj:
        merge_file_obj.truncate(0)
        for file_path_str in data_file_paths:
            print('\nReading file: ' + file_path_str + '\n')
            with open(file=file_path_str, mode='r') as read_file_obj:
                for i, line in enumerate(read_file_obj):
                    if i % verbose_period == 0:
                        print(i, '\t - ', line.strip())
                    if i == 10:
                        break

                    ts, sa_id, click, u_feats, aa_dict = parse_line(line=line)
                    # lines with user features that do not have specified number of features(=5 without biasses) are omitted
                    if len(u_feats) != 5:
                        print('line id: ', i, ' is omitted because its user features are:', u_feats)
                        continue

                    line_dict = {"timestamp": ts, "shown_article_id": sa_id, "click": click, "user_features": u_feats,
                                 "available_article_ids": list(aa_dict.keys())}
                    json_line_str = json.dumps(line_dict)
                    merge_file_obj.write(json_line_str + '\n')

                    art_old_size = len(article_dict)
                    article_dict.update(aa_dict)
                    art_new_size = len(article_dict)
                    if art_new_size != art_old_size:
                        print('New articles added, size of dict increase from {} to {}'.format(art_old_size, art_new_size))

    # articles that do not have specified number of features(=5 without biasses) are removed from article dict
    for a_id in list(article_dict.keys()):
        if len(article_dict[a_id]) != 5:
            a_f = article_dict.pop(a_id, None)
            print('article id: ', a_id, ' is omitted because its features are:', a_f)

    with open(article_dict_json_path, 'w') as json_file:
        json.dump(article_dict, json_file)


if __name__ == '__main__':
    data_files = OG_DATA_FILE_PATHS
    merge_data_into_single_file(data_file_paths=OG_DATA_FILE_PATHS, merge_file_path='demo_res_dat.dat',
                                article_dict_json_path='demo_art_dict.json', verbose_period=100000)
    # merge_data_into_single_file(data_file_paths=OG_DATA_FILE_PATHS, merge_file_path=RESTRUCTURED_DATA_FILE_PATH,
    #                             article_dict_json_path=ARTICLE_DICT_JSON_PATH, verbose_period=100000)