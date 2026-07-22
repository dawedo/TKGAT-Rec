import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from time import time


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data, user_item_adj, user_sequences = load_rating(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg(args)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation, user_item_adj, user_sequences


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)
    
    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    n_interactions = rating_np.shape[0]
    
    # NEW: Normalize timestamps to relative days
    # FIRST: Split the data into train/eval/test
    train_data, eval_data, test_data = dataset_split(rating_np, args)

    # THEN: Build sequences ONLY from training data (no leakage)
    if rating_np.shape[1] > 3:
        train_timestamps = train_data[:, 3]
        min_timestamp = train_timestamps.min()
        max_timestamp = train_timestamps.max()
        
        print(f"Timestamp range (train only): {min_timestamp} to {max_timestamp}")
        print(f"Date range: {pd.to_datetime(min_timestamp, unit='s')} to {pd.to_datetime(max_timestamp, unit='s')}")
        
        user_sequences = build_user_sequences_from_train(train_data, min_timestamp)
    else:
        user_sequences = build_user_sequences_from_train(train_data, 0)
    
    # train_data, eval_data, test_data = dataset_split(rating_np, args)
    # Verify sequences were built
    if user_sequences:
        seq_lengths = [len(seq) for seq in user_sequences.values() if len(seq) > 0]
        if seq_lengths:
            print(f"\n===== Sequence Quality Analysis =====")
            print(f"Users with sequences: {len([s for s in user_sequences.values() if len(s) > 0])}")
            print(f"Median sequence length: {np.median(seq_lengths):.1f}")
            print(f"Mean sequence length: {np.mean(seq_lengths):.1f}")
            print(f"Sequences < 5 items: {sum(1 for l in seq_lengths if l < 5)} / {len(seq_lengths)}")
            print(f"Sequences < 10 items: {sum(1 for l in seq_lengths if l < 10)} / {len(seq_lengths)}")
            print(f"Max sequence length: {max(seq_lengths)}")
            print("=====================================\n")
        else:
            print("WARNING: All sequences are empty!")
    
    # Build user sequences with temporal order
    # user_sequences = build_user_sequences(rating_np, args)
    
    # train_data, eval_data, test_data = dataset_split(rating_np, args)
    # ===== Additional statistics =====
    # user_sequences = {}
    # for u, i, r in train_data:
    #     if r > 0.5:  # only positive interactions
    #         if u not in user_sequences:
    #             user_sequences[u] = []
    #         user_sequences[u].append(i)

    valid_users = [u for u, seq in user_sequences.items() if len(seq) > 0]
    num_valid_users = len(valid_users)
    avg_seq_len = np.mean([len(seq) for seq in user_sequences.values()]) if num_valid_users > 0 else 0

    print("===== Dataset Statistics =====")
    print(f"Number of users: {n_user}")
    print(f"Number of items: {n_item}")
    print(f"Number of interactions: {n_interactions}")
    print(f"Users with valid sequences: {num_valid_users}")
    print(f"Average sequence length: {avg_seq_len:.2f}")
    print("==============================")

    '''
    *****************************************
    Rating Matrix for NGCF
    '''
    R = sp.dok_matrix((n_user, n_item), dtype=np.float32)
    for i in range(train_data.shape[0]):
        if train_data[i, 2] > 0.5:
            R[train_data[i, 0], train_data[i, 1]] = 1

    plain_adj, norm_adj, mean_adj = get_adj_mat(args, n_user, n_item, R)
    if args.adj_type == 'plain':
        user_item_adj = plain_adj
    elif args.adj_type == 'norm':
        user_item_adj = norm_adj
    elif args.adj_type == 'gcmc':
        user_item_adj = mean_adj
    else:
        user_item_adj = mean_adj + sp.eye(mean_adj.shape[0])
    '''
    *****************************************
    '''
    return n_user, n_item, train_data, eval_data, test_data, user_item_adj, user_sequences


def build_user_sequences_from_train(train_data, min_timestamp):
    """
    Build temporal sequences ONLY from training data
    Only positive interactions (rating > 0.5)
    Normalize timestamps to 0-9999 range for embedding lookup
    """
    user_sequences = {}
    
    # First pass: collect all sequences
    for row in train_data:
        user_id = int(row[0])
        item_id = int(row[1])
        rating = float(row[2])
        
        if rating <= 0.5:  # Skip negative samples
            continue
        
        # Get raw timestamp from column 3
        timestamp = float(row[3]) if len(row) > 3 else 0.0
        
        if user_id not in user_sequences:
            user_sequences[user_id] = []
        user_sequences[user_id].append((item_id, timestamp))
    
    # Find global min/max for normalization
    all_timestamps = []
    for seq in user_sequences.values():
        all_timestamps.extend([ts for _, ts in seq])
    
    if len(all_timestamps) > 0:
        global_min = min(all_timestamps)
        global_max = max(all_timestamps)
        timestamp_range = global_max - global_min
        
        print(f"Timestamp range: {global_min} to {global_max}")
        
        # Normalize to 0-9999 range
        for user_id in user_sequences:
            normalized_seq = []
            for item_id, timestamp in user_sequences[user_id]:
                if timestamp_range > 0:
                    # Normalize to 0-9999
                    normalized_ts = ((timestamp - global_min) / timestamp_range) * 9999.0
                else:
                    normalized_ts = 0.0
                normalized_seq.append((item_id, normalized_ts))
            
            # Sort by normalized timestamp
            normalized_seq = sorted(normalized_seq, key=lambda x: x[1])
            
            # Add position indices
            user_sequences[user_id] = [
                (item, ts, pos) 
                for pos, (item, ts) in enumerate(normalized_seq)
            ]
    
    print(f"Built sequences from training data only:")
    print(f"  Users with sequences: {len(user_sequences)}")
    if len(user_sequences) > 0:
        avg_len = np.mean([len(seq) for seq in user_sequences.values()])
        print(f"  Average sequence length: {avg_len:.2f}")
        
        # Show sample normalized timestamps
        sample_user = list(user_sequences.keys())[0]
        sample_timestamps = [ts for _, ts, _ in user_sequences[sample_user][:5]]
        print(f"  Sample normalized timestamps: {sample_timestamps}")
    
    return user_sequences


'''
*****************************************
From NGCF
'''
def get_adj_mat(args, n_user, n_item, R):
    path = '../data/{}'.format(args.dataset)
    print('Creating UI Graph ...')
    adj_mat, norm_adj_mat, mean_adj_mat = create_adj_mat(n_user, n_item, R)
    print('Finish Creating Adjacency Matrix of UI Graph.')

    return adj_mat, norm_adj_mat, mean_adj_mat


def create_adj_mat(n_user, n_item, R):
    t1 = time()
    adj_mat = sp.dok_matrix((n_user + n_item, n_user + n_item), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()

    adj_mat[:n_user, n_user:] = R
    adj_mat[n_user:, :n_user] = R.T
    adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape, time() - t1)

    t2 = time()

    def normalized_adj_single(adj):
        '''
        D^(-1)*A
        '''
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    mean_adj_mat = normalized_adj_single(adj_mat)

    print('already normalize adjacency matrix', time() - t2)
    return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


'''
*****************************************
'''

def dataset_split(rating_np, args):
    print('---------------')
    print(args.seed)
    print('---------------')
    np.random.seed(args.seed)

    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(args, kg, n_entity)

    return n_entity, n_relation, adj_entity, adj_relation


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def construct_adj(args, kg, entity_num):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation