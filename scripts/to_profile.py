from netam import experiment, rsmodels
from netam.common import pick_device
from netam.framework import load_shmoof_dataframes

expt = experiment.Experiment()

# shmoof_path = "/Users/matsen/data/shmoof_edges_11-Jan-2023_NoNode0_iqtree_K80+R_masked.csv"
shmoof_path = "/Users/matsen/data/shmoof_pcp_2023-11-30_MASKED.csv"
val_nickname = 'small'

train_df, val_df = load_shmoof_dataframes(shmoof_path, val_nickname=val_nickname) #, sample_count=1000)

train_data_by_kmer_length = expt.data_by_kmer_length_of(train_df)
val_data_by_kmer_length = expt.data_by_kmer_length_of(val_df)

train_data = train_data_by_kmer_length[3]
val_data = val_data_by_kmer_length[3]

model = rsmodels.RSCNNModel(
                kmer_length=3,
                kernel_size=9,
                embedding_dim=7,
                filter_count=16,
                dropout_prob=0.2,
            )
# train_data.to("cpu")
# val_data.to("cpu")
model.to(pick_device())

burrito = rsmodels.RSSHMBurrito(train_data, val_data, model)
burrito.train(epochs=10)
 