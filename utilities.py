

def replace_string_keys_with_int_keys(str_keyed_dict):
    int_keyed_dict = {int(key): item for key, item in str_keyed_dict.items()}
    return int_keyed_dict