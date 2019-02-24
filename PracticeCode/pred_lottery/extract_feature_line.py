import numpy as np
import pandas as pd


def search_num(data_array,
               search_range_left,
               search_range_right,
               middle_point_raw,
               col,
               pre_point_num):

    for s in range(search_range_left, search_range_right):

        try:
            if np.isnan(data_array[middle_point_raw, s]):
                continue
            else:
                new_num = int(s * 2) - col + 1
                if (new_num > 0) & (new_num <= data_array.shape[1]):
                    pre_point_num = np.append(pre_point_num, new_num)

        except IndexError:
            break
    return pre_point_num


def extract_feature(data_array, search_range_add1):

    search_range = search_range_add1 - 1

    search_range_half = int(search_range/2)

    num_raw, num_col = data_array.shape[0], data_array.shape[1]  # 10,12

    pre_point_num = []

    if num_raw % 2 == 0:
        raw_pointer = 0
    else:
        raw_pointer = 1

    for raw in range(raw_pointer, num_raw, 2):

        for col in range(num_col):
            # start_point_position = [raw, col]
            # start_point_num = data_array[raw, col]

            if np.isnan(data_array[raw, col]):
                continue

            else:

                middle_point_raw = int(num_raw - ((num_raw + 1 - raw) / 2) + 1)  # int 非常关键，否则为5.0

                # print(middle_point_raw)

                if col < search_range_half:
                    search_range_left = 0
                    search_range_right = col + search_range_half + 1

                    pre_point_num = search_num(data_array,
                                               search_range_left,
                                               search_range_right,
                                               middle_point_raw,
                                               col,
                                               pre_point_num)
                else:
                    if col >= (num_col - search_range_half):
                        search_range_left = col - search_range_half
                        search_range_right = num_col

                        pre_point_num = search_num(data_array,
                                                   search_range_left,
                                                   search_range_right,
                                                   middle_point_raw,
                                                   col,
                                                   pre_point_num)
                    else:
                        search_range_left = col - search_range_half
                        search_range_right = col + search_range_half + 1

                        pre_point_num = search_num(data_array,
                                                   search_range_left,
                                                   search_range_right,
                                                   middle_point_raw,
                                                   col,
                                                   pre_point_num)

    return pre_point_num


data_csv = pd.read_csv("C:/Users/Administrator/Desktop/test_1.csv", header=None)

data_ndarray = data_csv.values


''''''
result = extract_feature(data_ndarray, 9)

result_statics = pd.value_counts(result)

print(result_statics)











