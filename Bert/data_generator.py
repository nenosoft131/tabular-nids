import sys

import numpy as np
import os
from math import ceil
from sklearn.model_selection import train_test_split
from logger_config import get_logger
# from utils import Utils
from datasets import Dataset
import pandas as pd



def row_to_full_sentence(row):
    try:
        return (
            f"{row['PROTOCOL']} flow from {row['IPV4_SRC_ADDR']}:{row['L4_SRC_PORT']} "
            f"to {row['IPV4_DST_ADDR']}:{row['L4_DST_PORT']}, using L7 protocol {row['L7_PROTO']}. "
            f"Flow transferred {row['IN_BYTES']} bytes in ({row['IN_PKTS']} packets) and "
            f"{row['OUT_BYTES']} bytes out ({row['OUT_PKTS']} packets), lasting {row['FLOW_DURATION_MILLISECONDS']} ms. "
            f"Client duration: {row['DURATION_IN']} ms, Server duration: {row['DURATION_OUT']} ms. "
            f"TCP flags: {row['TCP_FLAGS']} (client: {row['CLIENT_TCP_FLAGS']}, server: {row['SERVER_TCP_FLAGS']}). "
            f"TTL range: {row['MIN_TTL']}–{row['MAX_TTL']}. Packet size range: {row['SHORTEST_FLOW_PKT']}–{row['LONGEST_FLOW_PKT']} bytes. "
            f"IP packet length range: {row['MIN_IP_PKT_LEN']}–{row['MAX_IP_PKT_LEN']} bytes. "
            f"Throughput: {row['SRC_TO_DST_SECOND_BYTES']} Bps src→dst, {row['DST_TO_SRC_SECOND_BYTES']} Bps dst→src. "
            f"Retransmissions: {row['RETRANSMITTED_IN_PKTS']} packets ({row['RETRANSMITTED_IN_BYTES']} bytes) src→dst, "
            f"{row['RETRANSMITTED_OUT_PKTS']} packets ({row['RETRANSMITTED_OUT_BYTES']} bytes) dst→src. "
            f"Avg throughput: {row['SRC_TO_DST_AVG_THROUGHPUT']} bps src→dst, {row['DST_TO_SRC_AVG_THROUGHPUT']} bps dst→src. "
            f"Packet size categories: ≤128: {row['NUM_PKTS_UP_TO_128_BYTES']}, 128–256: {row['NUM_PKTS_128_TO_256_BYTES']}, "
            f"256–512: {row['NUM_PKTS_256_TO_512_BYTES']}, 512–1024: {row['NUM_PKTS_512_TO_1024_BYTES']}, "
            f"1024–1514: {row['NUM_PKTS_1024_TO_1514_BYTES']}. "
            f"TCP win size: in={row['TCP_WIN_MAX_IN']}, out={row['TCP_WIN_MAX_OUT']}. "
            f"ICMP type: {row['ICMP_TYPE']} (IPv4: {row['ICMP_IPV4_TYPE']}). "
            f"DNS query ID: {row['DNS_QUERY_ID']}, type: {row['DNS_QUERY_TYPE']}, TTL: {row['DNS_TTL_ANSWER']}. "
            f"FTP return code: {row['FTP_COMMAND_RET_CODE']}."
        )
    except Exception as e:
        print(f"Error formatting row: {e}", file=sys.stderr)
        return f"Error formatting row: {e}"

class DataGenerator():
    def __init__(self,
                 seed: int=42,
                 dataset: str=None,
                 test_size: float=0.3,
                 generate_duplicates=False,
                 n_samples_lower_bound=1000,
                 n_samples_upper_bound=3000,
                 verbose=False):
        '''
        :param seed: seed for reproducible results
        :param dataset: specific the dataset name
        :param test_size: testing set size
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_lower_bound: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_lower_bound will be dropped
        :param n_samples_upper_bound: threshold for downsampling input samples, considering the computational cost
        '''

        self.seed = seed
        self.dataset = dataset
        self.test_size = test_size

        self.generate_duplicates = generate_duplicates
        self.n_samples_lower_bound = n_samples_lower_bound
        # self.n_samples_upper_bound = n_samples_upper_bound
        self.logger = get_logger(__name__)

        # myutils function
        # self.utils = Utils()

        self.verbose = verbose
        
    def generator(self,
                  X=None,
                  y=None,
                  la=None,
                  at_least_one_labeled=False,
                  meta=False):
        '''
        :param X: input X features
        :param y: input y labels
        :param la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        :param at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
        :param meta: whether to save the meta features extracted by the MetaOD method (see https://github.com/yzhao062/MetaOD)
        '''
        
        file_path = os.path.join(os.path.dirname(__file__), 'datasets', self.dataset + '.csv')
        
        df = pd.read_csv(file_path)
        # print(df)

        # set seed for reproducible results
        # self.utils.set_seed(self.seed)
        df['text'] = df.apply(row_to_full_sentence, axis=1)
        
        df_train_val, df_test = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=self.seed)
        df_train, df_val = train_test_split(df_train_val, test_size=0.25, stratify=df_train_val['Label'], random_state=self.seed)
        
        df_train = df_train.rename(columns={'Label': 'labels'})
        df_val = df_val.rename(columns={'Label': 'labels'})
        df_test = df_test.rename(columns={'Label': 'labels'})


        # ✅ Convert to Hugging Face Datasets
        train_ds = Dataset.from_pandas(df_train)
        val_ds = Dataset.from_pandas(df_val)
        test_ds = Dataset.from_pandas(df_test)
       
        return {'X_train': train_ds, 'X_val': val_ds,'X_test': test_ds}