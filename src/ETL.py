import numpy as np
import pandas as pd
import logging

pd.set_option('mode.chained_assignment', 'raise')
pd.options.display.max_columns = None
pd.options.mode.use_inf_as_na = True
logging.basicConfig(level=logging.INFO)


def read_file():
    """read file and preprocess dataset"""
    sample = pd.read_csv('../data/pit_sample_xnys.txt', sep='\t', skiprows=1)
    sample = sample[(sample.SecID.notna()) & (sample.IssID.notna())]  # filter out rows with missing identifiers
    sample.loc[:, ['SecID', 'IssID']] = sample.loc[:, ['SecID', 'IssID']].astype(int)  # convert id columns to int
    ids = ['IssuerName', 'Changed', 'SecID', 'IssID', 'CntryofIncorp', 'Isin', 'SIC', 'CIK', 'GICS', 'NAICS']
    sample = sample.loc[:,ids]
    return sample


def group_loader(sample, groupby_id):
    """generate group id value, and group index list"""    
    for groupby_id_value, group in sample.groupby(groupby_id):
        yield groupby_id_value, group.index.values.tolist()


def compare_id(row, groupby_id_value, groupby_id, ungroup_index_list, dict_value):
    """
    compare one row's groupby id value with a single group's groupby id value.
    if any of groupby id values in that row equals to single group's grouby id value, append rows index to that group's index list
    """
    if (row.name in ungroup_index_list) & (not row[groupby_id].isna().all()):
        equal_flag_list = []
        for i in range(len(groupby_id)):
            # if a specific row's group_id is not null and equals to group's group_id, return True  
            if (row.loc[groupby_id[i]] == np.nan) | (row.loc[groupby_id[i]] == groupby_id_value[i]):
                equal_flag_list.append(True) 
            else:
                equal_flag_list.append(False)
        # if all element in equal_flag_list is True, append the row's index into that group's index list
        if all(equal_flag_list):
            dict_value.append(row.name)
        else:
            dict_value = [row.name]
        ungroup_index_list.remove(row.name)


def arrange_groups(sample):
    """
    arrange groups on dataframe index level.
    return the dataframe reindexed by grouped index.
    """
    first_group_index_dict = {} # key: groupby_id_value (tuple), value: index_list (list)
    # group by ['SIC', 'GICS', 'NAICS'] on rows that do not contains null value in ['SIC', 'GICS', 'NAICS'] columns
    groupby_id = ['SIC', 'GICS', 'NAICS']
    sample_notna = sample[sample[groupby_id].notna().all(axis=1)]
    sample_contains_na = sample[~(sample[groupby_id].notna().all(axis=1))]
    for groupby_id_value, group_index_list in group_loader(sample=sample_notna,
                                                           groupby_id=groupby_id):
        first_group_index_dict[groupby_id_value] = group_index_list
    ungroup_index_list = sample_contains_na.index.values.tolist()
    logging.info(f'complete initial groupby operation based on SIC, GICS, NAICS. {len(ungroup_index_list)} element in list')
    
    for groupby_id_value, group_index_list in first_group_index_dict.items():
        sample_contains_na.apply(compare_id, axis=1,
                                 groupby_id_value=groupby_id_value,
                                 groupby_id=groupby_id, 
                                 ungroup_index_list=ungroup_index_list,
                                 dict_value=group_index_list)
    logging.info(f'complete compare_id operation. key: SIC, GICS, NAICS. {len(ungroup_index_list)} element in list')
    
    ###########################################################################################################################
    
    second_group_index_dict = {} # key: groupby_id_value, value: index_list
    groupby_id = ['CIK']
    sample_ungroup = sample.loc[ungroup_index_list,:]
    sample_notna = sample_ungroup[sample_ungroup[groupby_id].notna().all(axis=1)]
    sample_contains_na = sample_ungroup[~(sample_ungroup[groupby_id].notna().all(axis=1))]
    for groupby_id_value, group_index_list in group_loader(sample= sample_notna,
                                                           groupby_id=groupby_id):
        second_group_index_dict[groupby_id_value] = group_index_list
    ungroup_index_list = sample_contains_na.index.values.tolist()
    logging.info(f'complete groupby operation with remaining records based on CIK. {len(ungroup_index_list)} element in list')
    
    
    for groupby_id_value, group_index_list in second_group_index_dict.items():
        sample_contains_na.apply(compare_id, axis=1,
                                 groupby_id_value=groupby_id_value,
                                 groupby_id=groupby_id,
                                 ungroup_index_list=ungroup_index_list, 
                                 dict_value=group_index_list)
    logging.info(f'complete compare_id operation. key: CIK. {len(ungroup_index_list)} element in list')
    
    [ungroup_index_list.extend(l) for l in list(first_group_index_dict.values())]
    [ungroup_index_list.extend(l) for l in list(second_group_index_dict.values())]
    logging.info(f'complete join operation on index. {len(ungroup_index_list)} element in list')
    
    sample_group = sample.reindex(ungroup_index_list)
    return sample_group


def write_output(sample_group):
    """write dataframe into csv file format in output folder"""
    sample_group.to_csv('../output/joined_Sec.csv', index=False)


def main():
    sample = read_file()
    sample_group = arrange_groups(sample)
    write_output(sample_group)


if __name__ == '__main__':
    main()