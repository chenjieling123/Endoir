# -*- coding:utf-8 -*-
# @FileName  :create_data_lists.py
# @Time      :2023/10/31 10:09
# @Author    :CJL


from utils import create_data_lists

if __name__ == '__main__':
    #create_data_lists(train_folders=["./garnn-rawData/tissue_train"],test_folders=["./garnn-rawData/tissue_test"],min_size=100,output_folder='./garnn-rawData/tissue')

    create_data_lists(train_folders=["./data"],test_folders=["./data"],min_size=100,output_folder='./data')

    '''
    create_data_lists(train_folders=['./garnn-rawData/kidney/test001',
                                     './garnn-rawData/kidney/test002',
                                     './garnn-rawData/kidney/test003',
                                     './garnn-rawData/kidney/test004',
                                     './garnn-rawData/liver/test001',
                                     './garnn-rawData/liver/test002',
                                     './garnn-rawData/liver/test003',
                                     './garnn-rawData/liver/test004',
                                     './garnn-rawData/tonsil/test001',
                                     './garnn-rawData/tonsil/test002',
                                     './garnn-rawData/tonsil/test003',
                                     './garnn-rawData/tonsil/test004'],
                      test_folders=[],
                      min_size=100,
                      output_folder='./garnn-rawData/histological')
    '''
