import tensorflow as tf
import numpy as np
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score


class KGCN(object):
    def __init__(self, args, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation, user_item_adj, user_sequences, pretrain=None):
        self.pretrain = pretrain
        self.user_sequences = user_sequences  # NEW
        self._parse_args(args, n_user, n_item, adj_entity, adj_relation, user_item_adj)
        self._build_inputs()
        self._build_model(n_user, n_item, n_entity, n_relation)
        self._build_train()

    tf.compat.v1.disable_eager_execution()

    @staticmethod
    def get_initializer():
        return tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

    def _parse_args(self, args, n_user, n_item, adj_entity, adj_relation, user_item_adj):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
#        self.layer_cl = args.layer_cl
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)
        # NEW: Temporal and attention parameters
        self.use_temporal = args.use_temporal if hasattr(args, 'use_temporal') else False
        self.max_seq_length = args.max_seq_length if hasattr(args, 'max_seq_length') else 50
        self.num_heads = args.num_heads if hasattr(args, 'num_heads') else 2
        self.seq_weight = args.seq_weight if hasattr(args, 'seq_weight') else 0.1

        '''
        *****************************************
        CF parameters
        '''
        self.n_user = n_user
        self.n_item = n_item
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.model_type = args.model_type
        self.smoothing_steps = args.smoothing_steps
        self.norm_adj = user_item_adj  # To do
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag

        # Create Model Parameters (i.e., Initialize Weights).
        if self.model_type in ['KGCN_NGCF', 'KGCN_LightGCN', 'NGCF', 'LightGCN']:
            self.weights_ngcf = self._init_weights_ngcf()

        self.att = args.att  # or 'u_r'
        if self.att in ['h_ur_t', 'h_r_t', 'hrt_add', 'ur_ht_mlp', 'u_r_mlp', 'uhrt_concat', 'uhrt_add',
                        'uhrt_add_2', 'uhrt_bi', 'u_h_r_t_mlp']:
            self.weights_att = self._init_weights_att()
            self.weights_att['att'] = self.att
        else:
            self.weights_att = dict()
            self.weights_att['att'] = self.att

        # item embeddings aggregation from KGCN and CF
        self.agg_type = args.agg_type
        self.alpha = args.alpha
#        self.beta = args.beta
        if self.agg_type in ['gcn', 'graphsage', 'bi']:
            self.weights_agg = self._init_weights_agg()
        
        
        # NEW: Temporal parameters
        self.max_seq_length = args.max_seq_length if hasattr(args, 'max_seq_length') else 50
        self.num_heads = args.num_heads if hasattr(args, 'num_heads') else 2
        self.use_temporal = args.use_temporal if hasattr(args, 'use_temporal') else True
        
        '''
        *****************************************
        '''

    def _build_inputs(self):
        self.user_indices = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None],
                                                     name='user_indices_{}'.format(self.model_type))
        self.item_indices = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None],
                                                     name='item_indices_{}'.format(self.model_type))
        self.labels = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='labels_{}'.format(self.model_type))

        # NEW: Temporal sequence inputs
        self.user_seq_items = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None, self.max_seq_length], name='user_seq_items')
        self.user_seq_timestamps = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.max_seq_length], name='user_seq_timestamps')
        self.user_seq_positions = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None, self.max_seq_length], name='user_seq_positions')
        self.user_seq_length = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None], name='user_seq_length')
        
        '''
        *****************************************
        NGCF dropout
        '''
        self.node_dropout = tf.compat.v1.placeholder(tf.float32, shape=[None],
                                                     name='node_dropout_{}'.format(self.model_type))
        self.mess_dropout = tf.compat.v1.placeholder(tf.float32, shape=[None],
                                                     name='mess_dropout_{}'.format(self.model_type))
        
        '''
        *****************************************
        '''

    
    def _build_model(self, n_user, n_item, n_entity, n_relation):
        with tf.compat.v1.variable_scope('{}'.format(self.model_type)):
            if self.pretrain is not None:
                self.user_emb_matrix = tf.compat.v1.get_variable(initializer=self.pretrain['user'],
                                                                 name='user_emb_matrix_{}'.format(self.model_type))
                self.entity_emb_matrix = tf.compat.v1.get_variable(initializer=self.pretrain['entity'],
                                                                   name='entity_emb_matrix_{}'.format(self.model_type))
                self.relation_emb_matrix = tf.compat.v1.get_variable(initializer=self.pretrain['relation'],
                                                                     name='relation_emb_matrix_{}'.format(
                                                                         self.model_type))
            else:
                self.user_emb_matrix = tf.compat.v1.get_variable(
                    shape=[n_user, self.dim], initializer=KGCN.get_initializer(),
                    name='user_emb_matrix_{}'.format(self.model_type))
                # self.item_emb_matrix = tf.get_variable(
                #     shape=[n_item, self.dim], initializer=KGCN.get_initializer(), name='item_emb_matrix_{}'.format(self.model_type))
                self.entity_emb_matrix = tf.compat.v1.get_variable(
                    shape=[n_entity, self.dim], initializer=KGCN.get_initializer(),
                    name='entity_emb_matrix_{}'.format(self.model_type))
                self.relation_emb_matrix = tf.compat.v1.get_variable(
                    shape=[n_relation, self.dim], initializer=KGCN.get_initializer(),
                    name='relation_emb_matrix_{}'.format(self.model_type))
            
            # Learnable weights for path-aware prediction
            self.context_weight = tf.compat.v1.get_variable(
                shape=[1],
                initializer=tf.constant_initializer(0.3),
                name='context_weight_{}'.format(self.model_type)
            )

            self.path_weights = tf.compat.v1.get_variable(
                shape=[3],
                initializer=tf.constant_initializer([0.5, 0.3, 0.2]),
                name='path_weights_{}'.format(self.model_type)
            )      
            # NEW: Add temporal and positional embeddings
            if self.use_temporal:
                
                self.temporal_emb_matrix = tf.compat.v1.get_variable(
                    shape=[10000, self.dim],  # Keep 10000 or adjust based on normalized range
                    initializer=KGCN.get_initializer(),
                    name='temporal_emb_matrix_{}'.format(self.model_type)
                )
                
                # Positional embeddings for sequence position
                self.position_emb_matrix = tf.compat.v1.get_variable(
                    shape=[self.max_seq_length, self.dim],
                    initializer=KGCN.get_initializer(),
                    name='position_emb_matrix_{}'.format(self.model_type)
                )
                
                # Attention weights (Q, K, V matrices)
                self.W_Q = tf.compat.v1.get_variable(
                    shape=[self.dim, self.dim],
                    initializer=KGCN.get_initializer(),
                    name='W_Q_{}'.format(self.model_type)
                )
                self.W_K = tf.compat.v1.get_variable(
                    shape=[self.dim, self.dim],
                    initializer=KGCN.get_initializer(),
                    name='W_K_{}'.format(self.model_type)
                )
                self.W_V = tf.compat.v1.get_variable(
                    shape=[self.dim, self.dim],
                    initializer=KGCN.get_initializer(),
                    name='W_V_{}'.format(self.model_type)
                )
                
                # Gate fusion weights
                self.W_gate = tf.compat.v1.get_variable(
                    shape=[self.dim, self.dim],
                    initializer=KGCN.get_initializer(),
                    name='W_gate_{}'.format(self.model_type)
                )
                self.b_gate = tf.compat.v1.get_variable(
                    shape=[self.dim],
                    initializer=tf.zeros_initializer(),
                    name='b_gate_{}'.format(self.model_type)
                )
                # ✅ ADD THIS
                self.W_O = tf.compat.v1.get_variable(
                    shape=[self.dim, self.dim],
                    initializer=KGCN.get_initializer(),
                    name='W_O_{}'.format(self.model_type)
                )

        '''
        *****************************************
        User Embedding
        '''
        # self.user_embeddings_kg = tf.nn.embedding_lookup(params=self.user_emb_matrix, ids=self.user_indices)
        # WITH this logic:
        if self.model_type in ['KGCN_NGCF', 'KGCN_LightGCN']:
            # First create LightGCN embeddings
            if self.alg_type == 'lightgcn':
                self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()
            
            # Use LightGCN output as KG embeddings for temporal fusion
            self.user_embeddings_kg = tf.nn.embedding_lookup(params=self.ua_embeddings, ids=self.user_indices)
        else:
            # For pure KGCN, use initial embeddings
            self.user_embeddings_kg = tf.nn.embedding_lookup(params=self.user_emb_matrix, ids=self.user_indices)
        # NEW: Create sequence-based embeddings if temporal mode is enabled
        if self.use_temporal:
            print("Using temporal sequences with cross-attention")
            # Get sequence representations (with positional/temporal info)
            seq_item_embs = tf.nn.embedding_lookup(params=self.entity_emb_matrix, ids=self.user_seq_items)
            # if self.model_type in ['KGCN_LightGCN', 'LightGCN']:
            #     seq_item_embs = tf.nn.embedding_lookup(params=self.ia_embeddings, ids=self.user_seq_items)
            # else:
            #     seq_item_embs = tf.nn.embedding_lookup(params=self.entity_emb_matrix, ids=self.user_seq_items)
           
            temporal_indices = tf.cast(tf.minimum(self.user_seq_timestamps, 9999.0), tf.int64)
            seq_temporal_embs = tf.nn.embedding_lookup(params=self.temporal_emb_matrix, ids=temporal_indices)
            seq_position_embs = tf.nn.embedding_lookup(params=self.position_emb_matrix, ids=self.user_seq_positions)
            combined_seq_embs = seq_item_embs + seq_temporal_embs + seq_position_embs
            
            # Cross-attention: KG embeddings attend to sequence
            self.cross_attention_output = self._cross_attention(self.user_embeddings_kg, combined_seq_embs)
            
            # Self-attention on sequences
            self.user_embeddings_seq = self._create_sequence_embeddings()
            
            # Gate fusion: combine KG + cross-attention + self-attention
            combined_context = self.cross_attention_output + self.user_embeddings_seq
            self.user_embeddings_final = self._gate_fusion(self.user_embeddings_kg, combined_context)
           
        else:
            print("✗ Using only KG-based user embeddings (no temporal)")
            self.user_embeddings_final = self.user_embeddings_kg

        if self.model_type == 'KGCN':
            
            entities, relations = self.get_neighbors(self.item_indices)
            # [batch_size, dim]
            self.item_embeddings_final, self.aggregators = self.aggregate(entities, relations)

        elif self.model_type in ['NGCF', 'LightGCN']:
            # ngcf, gcn or gcmc for user and item embeddings
            if self.alg_type == 'ngcf':
                self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
            elif self.alg_type == 'lightgcn':
                self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()
            else:
                raise Exception("Unknown alg_type: " + self.alg_type)

            # Only use CF user embeddings if temporal is disabled
            if not self.use_temporal:
                self.user_embeddings_final = tf.nn.embedding_lookup(params=self.ua_embeddings, ids=self.user_indices)
            # else: keep the temporal+KG fused embeddings from above
            
            self.item_embeddings_final = tf.nn.embedding_lookup(params=self.ia_embeddings, ids=self.item_indices)
            

        elif self.model_type in ['KGCN_NGCF', 'KGCN_LightGCN']:
            # ngcf, gcn or gcmc for user embeddings
            if self.alg_type == 'ngcf':
                self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
            elif self.alg_type == 'lightgcn':
                self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()
            else:
                raise Exception("Unknown alg_type: " + self.alg_type)
            # Only use CF user embeddings if temporal is disabled
            if not self.use_temporal:
                self.user_embeddings_final = tf.nn.embedding_lookup(params=self.ua_embeddings, ids=self.user_indices)
            # else: keep the temporal+KG fused embeddings from above
            
            self.item_embeddings_cf = tf.nn.embedding_lookup(params=self.ia_embeddings, ids=self.item_indices)

            # KGCN for item embeddings
            entities, relations = self.get_neighbors(self.item_indices)
            self.item_embeddings_kg, self.aggregators = self.aggregate(entities, relations)

           

            # combine item embeddings from CF & KG
            if self.agg_type == 'weighted_avg':
                self.item_embeddings_final = self.alpha * self.item_embeddings_cf + (
                            1 - self.alpha) * self.item_embeddings_kg
            elif self.agg_type == 'gcn':
                # item_embeddings = self.alpha*self.item_embeddings_cf + (1-self.alpha)*self.item_embeddings_kg
                item_embeddings = self.item_embeddings_cf + self.item_embeddings_kg
                self.item_embeddings_final = tf.nn.leaky_relu(tf.matmul(item_embeddings, self.weights_agg['agg_w_1']))
            elif self.agg_type == 'graphsage':
                item_embeddings = tf.concat([self.item_embeddings_cf, self.item_embeddings_kg], axis=-1)
                self.item_embeddings_final = tf.nn.leaky_relu(tf.matmul(item_embeddings, self.weights_agg['agg_w_1']))
            elif self.agg_type == 'bi':
                # item_embeddings_1 = self.alpha*self.item_embeddings_cf + (1-self.alpha)*self.item_embeddings_kg
                item_embeddings_1 = self.item_embeddings_cf + self.item_embeddings_kg
                item_embeddings_2 = tf.multiply(self.item_embeddings_cf, self.item_embeddings_kg)
                self.item_embeddings_final = tf.nn.leaky_relu(
                    tf.matmul(item_embeddings_1, self.weights_agg['agg_w_1'])) + \
                                             tf.nn.leaky_relu(tf.matmul(item_embeddings_2, self.weights_agg['agg_w_2']))

            else:
                raise Exception("Unknown model_type: " + self.model_type)
        '''
        *****************************************
        '''

        # [batch_size]old
        # self.scores = tf.reduce_sum(input_tensor=self.user_embeddings_final * self.item_embeddings_final, axis=1)
        # self.scores_normalized = tf.sigmoid(self.scores)
        # NEW:
        self.scores = self._path_aware_prediction(self.user_embeddings_final, self.item_embeddings_final)
        self.scores_normalized = tf.sigmoid(self.scores)
        
   
    def _path_aware_prediction(self, user_emb, item_emb):
        """Enhanced path-aware prediction with LEARNABLE weights"""
        
        # 1-hop neighbors
        item_neighbors_1hop = tf.nn.embedding_lookup(self.adj_entity, self.item_indices)
        
        # ✅ Use the SAME embedding source as item_emb for consistency
        if self.model_type in ['KGCN_NGCF', 'KGCN_LightGCN']:
            # For hybrid models, use CF embeddings (ia_embeddings) for neighbors
            neighbor_embs_1hop = tf.nn.embedding_lookup(self.ia_embeddings, item_neighbors_1hop)
        else:
            # For pure KGCN, use entity embeddings
            neighbor_embs_1hop = tf.nn.embedding_lookup(self.entity_emb_matrix, item_neighbors_1hop)
        
        # Aggregate neighbor context
        neighbor_context_1hop = tf.reduce_mean(neighbor_embs_1hop, axis=1)
        
        # Path 1: Base score (user-item direct)
        base_score = tf.reduce_sum(user_emb * item_emb, axis=1)
        
        # Path 2: Path score (user-neighborhood)
        path_score_1hop = tf.reduce_sum(user_emb * neighbor_context_1hop, axis=1)
        
        # Path 3: Context-aware score with LEARNABLE weight
        item_with_context = item_emb + self.context_weight * neighbor_context_1hop
        context_score = tf.reduce_sum(user_emb * item_with_context, axis=1)
        
        # Combine with LEARNABLE weights
        path_weights = tf.nn.softmax(self.path_weights)
        
        final_score = (path_weights[0] * base_score + 
                    path_weights[1] * path_score_1hop + 
                    path_weights[2] * context_score)
        
        return final_score
    '''
    *****************************************
    weights initialization for attention
    '''

    def _init_weights_att(self):
        all_weights = dict()

        if self.att == 'h_ur_t':
            shape_w_1 = [3 * self.dim, 1]

        elif self.att == 'h_r_t':
            shape_w_1 = [3 * self.dim, 1]

        elif self.att == 'hrt_add':
            shape_w_1 = [self.dim, 1]

        elif self.att == 'ur_ht_mlp':
            shape_w_1 = [2 * self.dim, self.dim]
            shape_w_2 = [self.dim, 1]

        elif self.att in ['u_r_mlp']:
            shape_w_1 = [2 * self.dim, self.dim]
            shape_w_2 = [self.dim, 1]

        elif self.att in ['u_h_r_t_mlp']:
            shape_w_1 = [2 * self.dim, self.dim]
            shape_w_2 = [self.dim, 1]

            shape_w_1_2 = [2 * self.dim, self.dim]
            shape_w_2_2 = [self.dim, 1]

        elif self.att in ['uhrt_concat']:
            shape_w_1 = [4 * self.dim, 1]

        elif self.att in ['uhrt_add']:
            shape_w_1 = [self.dim, 1]

        elif self.att in ['uhrt_add_2']:
            shape_w_1 = [self.dim, self.dim]
            shape_w_2 = [self.dim, 1]

        elif self.att == 'uhrt_bi':
            shape_w_1 = [self.dim, self.dim]
            shape_w_2 = [self.dim, self.dim]
            shape_w_3 = [self.dim, 1]

        with tf.compat.v1.variable_scope('{}'.format(self.model_type)):

            if self.att in ['h_r_t', 'h_ur_t', 'uhrt_concat', 'uhrt_add']:
                all_weights['att_w_1'] = tf.compat.v1.get_variable(
                    shape=shape_w_1, initializer=KGCN.get_initializer(), name='att_w_1_{}'.format(self.model_type))
                all_weights['att_b_1'] = tf.compat.v1.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_1_{}'.format(self.model_type))

            if self.att in ['uhrt_bi']:
                all_weights['att_w_1'] = tf.compat.v1.get_variable(
                    shape=shape_w_1, initializer=KGCN.get_initializer(), name='att_w_1_{}'.format(self.model_type))
                all_weights['att_b_1'] = tf.compat.v1.get_variable(
                    shape=[1, self.dim], initializer=KGCN.get_initializer(), name='att_b_1_{}'.format(self.model_type))

                all_weights['att_w_2'] = tf.compat.v1.get_variable(
                    shape=shape_w_2, initializer=KGCN.get_initializer(), name='att_w_2_{}'.format(self.model_type))
                all_weights['att_b_2'] = tf.compat.v1.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_2_{}'.format(self.model_type))

                all_weights['att_w_3'] = tf.compat.v1.get_variable(
                    shape=shape_w_3, initializer=KGCN.get_initializer(), name='att_w_3_{}'.format(self.model_type))
                all_weights['att_b_3'] = tf.compat.v1.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_3_{}'.format(self.model_type))

            if self.att in ['ur_ht_mlp', 'u_r_mlp', 'uhrt_add_2']:
                all_weights['att_w_1'] = tf.compat.v1.get_variable(
                    shape=shape_w_1, initializer=KGCN.get_initializer(), name='att_w_1_{}'.format(self.model_type))
                all_weights['att_b_1'] = tf.compat.v1.get_variable(
                    shape=[1, self.dim], initializer=KGCN.get_initializer(), name='att_b_1_{}'.format(self.model_type))

                all_weights['att_w_2'] = tf.compat.v1.get_variable(
                    shape=shape_w_2, initializer=KGCN.get_initializer(), name='att_w_2_{}'.format(self.model_type))
                all_weights['att_b_2'] = tf.compat.v1.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_2_{}'.format(self.model_type))

            if self.att in ['u_h_r_t_mlp']:
                all_weights['att_w_1'] = tf.compat.v1.get_variable(
                    shape=shape_w_1, initializer=KGCN.get_initializer(), name='att_w_1_{}'.format(self.model_type))
                all_weights['att_b_1'] = tf.compat.v1.get_variable(
                    shape=[1, self.dim], initializer=KGCN.get_initializer(), name='att_b_1_{}'.format(self.model_type))

                all_weights['att_w_2'] = tf.compat.v1.get_variable(
                    shape=shape_w_2, initializer=KGCN.get_initializer(), name='att_w_2_{}'.format(self.model_type))
                all_weights['att_b_2'] = tf.compat.v1.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_2_{}'.format(self.model_type))

                all_weights['att_w_1_2'] = tf.compat.v1.get_variable(
                    shape=shape_w_1_2, initializer=KGCN.get_initializer(), name='att_w_1_2_{}'.format(self.model_type))
                all_weights['att_b_1_2'] = tf.compat.v1.get_variable(
                    shape=[1, self.dim], initializer=KGCN.get_initializer(), name='att_b_1_2{}'.format(self.model_type))

                all_weights['att_w_2_2'] = tf.compat.v1.get_variable(
                    shape=shape_w_2_2, initializer=KGCN.get_initializer(), name='att_w_2_2_{}'.format(self.model_type))
                all_weights['att_b_2_2'] = tf.compat.v1.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_2_2_{}'.format(self.model_type))

        return all_weights

    '''
    *****************************************
    '''

    '''
    *****************************************
    weights initialization for combining item embeddings
    '''

    def _init_weights_agg(self):
        all_weights = dict()

        if self.agg_type in ['gcn', 'bi']:
            shape_w = [self.dim, self.dim]
            shape_b = [1, self.dim]
        elif self.agg_type == 'graphsage':
            shape_w = [2 * self.dim, self.dim]
            shape_b = [1, self.dim]
        else:
            raise Exception('Unknown  agg_type: {}'.format(self.agg_type))

        with tf.compat.v1.variable_scope('{}'.format(self.model_type)):
            all_weights['agg_w_1'] = tf.compat.v1.get_variable(
                shape=shape_w, initializer=KGCN.get_initializer(), name='agg_w_1_{}'.format(self.model_type))
            all_weights['agg_b_1'] = tf.compat.v1.get_variable(
                shape=shape_b, initializer=KGCN.get_initializer(), name='agg_b_1_{}'.format(self.model_type))

            if self.agg_type == 'bi':
                all_weights['agg_w_2'] = tf.compat.v1.get_variable(
                    shape=shape_w, initializer=KGCN.get_initializer(), name='agg_w_2_{}'.format(self.model_type))
                all_weights['agg_b_2'] = tf.compat.v1.get_variable(
                    shape=shape_b, initializer=KGCN.get_initializer(), name='agg_b_2_{}'.format(self.model_type))

        return all_weights

    '''
    *****************************************
    '''

    '''
    *****************************************
    user and item embedding with NGCF
    '''

    def _init_weights_ngcf(self):
        all_weights = dict()

        self.weight_size_list = [self.dim] + self.weight_size

        for k in range(self.n_layers):

            with tf.compat.v1.variable_scope('{}'.format(self.model_type)):

                all_weights['W_gc_%d' % k] = tf.compat.v1.get_variable(
                    shape=[self.weight_size_list[k], self.weight_size_list[k + 1]], initializer=KGCN.get_initializer(),
                    name='W_gc_{}_{}'.format(k, self.model_type))
                all_weights['b_gc_%d' % k] = tf.compat.v1.get_variable(
                    shape=[1, self.weight_size_list[k + 1]], initializer=KGCN.get_initializer(),
                    name='b_gc_{}_{}'.format(k, self.model_type))

                if self.alg_type == 'ngcf':
                    all_weights['W_bi_%d' % k] = tf.compat.v1.get_variable(
                        shape=[self.weight_size_list[k], self.weight_size_list[k + 1]],
                        initializer=KGCN.get_initializer(), name='W_bi_{}_{}'.format(k, self.model_type))
                    all_weights['b_bi_%d' % k] = tf.compat.v1.get_variable(
                        shape=[1, self.weight_size_list[k + 1]], initializer=KGCN.get_initializer(),
                        name='b_bi_{}_{}'.format(k, self.model_type))

                if self.alg_type == 'lightgcn':
                    all_weights['W_lt_%d' % k] = tf.compat.v1.get_variable(
                        shape=[self.weight_size_list[k], self.weight_size_list[k + 1]],
                        initializer=KGCN.get_initializer(), name='W_lt_{}_{}'.format(k, self.model_type))
                    all_weights['b_lt_%d' % k] = tf.compat.v1.get_variable(
                        shape=[1, self.weight_size_list[k + 1]], initializer=KGCN.get_initializer(),
                        name='b_lt_{}_{}'.format(k, self.model_type))

        return all_weights

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random.uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse.retain(X, dropout_mask)

        return pre_out * tf.compat.v1.div(1., keep_prob)

    def _split_A_hat(self, X):
        A_fold_hat = []
        n_fold = 100
        fold_len = (self.n_user + self.n_item) // n_fold
        for i_fold in range(n_fold):
            start = i_fold * fold_len
            if i_fold == n_fold - 1:
                end = self.n_user + self.n_item
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat


    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []
        n_fold = 100
        fold_len = (self.n_user + self.n_item) // n_fold
        for i_fold in range(n_fold):
            start = i_fold * fold_len
            if i_fold == n_fold - 1:
                end = self.n_user + self.n_item
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_lightgcn_embed(self):
        n_fold = 100
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.user_emb_matrix, self.entity_emb_matrix[:self.n_item, :]], axis=0)  # E
        #ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(n_fold):
                temp_embed.append(tf.sparse.sparse_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_user, self.n_item], 0)
        return u_g_embeddings, i_g_embeddings


    def _create_ngcf_embed(self):
        if self.node_dropout_flag:
            # node dropout.
            temp = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
            A = self._dropout_sparse(temp, 1 - self.node_dropout[0], self.n_nonzero_elems)
        else:
            A = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        ego_embeddings = tf.concat([self.user_emb_matrix, self.entity_emb_matrix[:self.n_item, :]], axis=0)  # E
        # ego_embeddings = tf.concat([self.user_emb_matrix, self.item_emb_matrix], axis=0)  # E

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            side_embeddings = tf.sparse.sparse_dense_matmul(A, ego_embeddings)  # Eq. (7) L*E
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights_ngcf['W_gc_%d' % k]) + self.weights_ngcf[
                    'b_gc_%d' % k])  # Eq. (7) L*E*W

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)  # Eq. (7) LE element_wise_dot E
            # transformed bi messages of neighbors.
            # Eq. (7) LE element_wise_dot E*W
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights_ngcf['W_bi_%d' % k]) + self.weights_ngcf['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, rate=1 - (1 - self.mess_dropout[k]))

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]
        # Method 1
        all_embeddings = tf.concat(all_embeddings, 1)  # Eq. (9)
        # # Method 2
        # all_embeddings = all_embeddings[-1]
        # # Method 3
        # all_embeddings = tf.add_n(all_embeddings)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_user, self.n_item], 0)
        return u_g_embeddings, i_g_embeddings


        '''
    *****************************************
    '''

    '''
    ******************************************
    item embedding in knowledge graph with KGCN
    '''

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(params=self.entity_emb_matrix, ids=i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(params=self.relation_emb_matrix, ids=i) for i in relations]

        res = [tf.reshape(entity_vectors[0], [self.batch_size, self.dim])]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.weights_att, self.batch_size, self.dim,
                                                   self.n_neighbor, act=tf.nn.tanh,
                                                   name=self.model_type + '_{}'.format(i))
            else:
                aggregator = self.aggregator_class(self.weights_att, self.batch_size, self.dim,
                                                   self.n_neighbor, name=self.model_type + '_{}'.format(i))
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    # user_embeddings=self.user_embeddings,
                                    user_embeddings=self.user_embeddings_kg, #was user_embeddings
                                    masks=None,
                                    hops=hop + 1)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
            res.append(tf.reshape(entity_vectors[0], [self.batch_size, self.dim]))

        # res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        # # Method 1
        # res = tf.concat(res, 1)

        # Method 2
        res = tf.add_n(res)

        # # Method 3
        # res = res[-1]

        return res, aggregators

    '''
    *****************************************
    '''

    def _build_train(self):
        self.base_loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        # self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(self.entity_emb_matrix) + \
        #         tf.nn.l2_loss(self.relation_emb_matrix) + tf.nn.l2_loss(self.item_emb_matrix)

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(self.entity_emb_matrix) + \
                       tf.nn.l2_loss(self.relation_emb_matrix)
        self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.context_weight) + tf.nn.l2_loss(self.path_weights)
        # NEW: Add L2 loss for temporal components if enabled
        
        if self.use_temporal:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.temporal_emb_matrix) + \
                    tf.nn.l2_loss(self.position_emb_matrix) + \
                    tf.nn.l2_loss(self.W_Q) + tf.nn.l2_loss(self.W_K) + tf.nn.l2_loss(self.W_V) + \
                    tf.nn.l2_loss(self.W_gate) + \
                    tf.nn.l2_loss(self.W_O)
            # ADD SEQUENTIAL LOSS HERE
            self.seq_loss = self._build_sequential_loss()
        else:
            self.seq_loss = tf.constant(0.0)

        if self.model_type in ['KGCN', 'KGCN_NGCF', 'KGCN_LightGCN']:
            for aggregator in self.aggregators:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

            if self.weights_att['att'] in ['h_ur_t', 'h_r_t' 'ur_ht_mlp', 'u_r_mlp', 'uhrt_concat', 'uhrt_add',
                                           'uhrt_add_2', 'uhrt_bi', 'u_h_r_t_mlp']:
                for weights in self.weights_att.keys():
                    if weights.startswith('att_w'):
                        self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_att[weights])

            if self.agg_type in ['gcn', 'graphsage']:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_agg['agg_w_1'])

            if self.agg_type == 'bi':
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_agg['agg_w_1']) + \
                               tf.nn.l2_loss(self.weights_agg['agg_w_2'])
        # MODIFIED: Combine all losses with weights
        # seq_weight controls the importance of sequential prediction
        # self.seq_weight = args.seq_weight # You can make this a hyperparameter
        
        self.loss = self.base_loss + self.l2_weight * self.l2_loss
        
        if self.use_temporal:
            self.loss = self.loss + self.seq_weight * self.seq_loss
        
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

        '''
        ******************************************
        L2 loss in NGCF
        '''
        # l2_loss_ngcf = []
        # for key in self.weights_ngcf:
        #     if 'W' in key:
        #         l2_loss_ngcf.append(tf.nn.l2_loss(self.weights_ngcf[key]))
        # l2_loss_ngcf = tf.add_n(l2_loss_ngcf)

        # self.l2_loss = self.l2_loss + l2_loss_ngcf
        '''
        *****************************************
        '''

        # self.loss = self.base_loss + self.l2_weight * self.l2_loss
        # self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.base_loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)

    def get_embeddings(self, sess):
        user_emb, entity_emb, relation_emb = sess.run(
            [self.user_emb_matrix, self.entity_emb_matrix, self.relation_emb_matrix])

        return user_emb, entity_emb, relation_emb
    # ADD THESE NEW METHODS TO THE KGCN CLASS

    def _create_sequence_embeddings(self):
        """Just a wrapper now"""
        seq_item_embs = tf.nn.embedding_lookup(params=self.entity_emb_matrix, ids=self.user_seq_items)
        # # # Get raw sequence embeddings
        # if self.model_type in ['KGCN_LightGCN', 'LightGCN']:
        #     seq_item_embs = tf.nn.embedding_lookup(params=self.ia_embeddings, ids=self.user_seq_items)
        # else:
        #     seq_item_embs = tf.nn.embedding_lookup(params=self.entity_emb_matrix, ids=self.user_seq_items)
        
        temporal_indices = tf.cast(tf.minimum(self.user_seq_timestamps, 9999.0), tf.int64)
        seq_temporal_embs = tf.nn.embedding_lookup(params=self.temporal_emb_matrix, ids=temporal_indices)
        seq_position_embs = tf.nn.embedding_lookup(params=self.position_emb_matrix, ids=self.user_seq_positions)
        
        combined_embs = seq_item_embs + seq_temporal_embs + seq_position_embs
        
        # Apply self-attention and pool
        # self_attended = self._multi_head_self_attention(combined_embs)
        self_attended = self._multi_head_attention(combined_embs)
        return self._pool_sequence(self_attended)
        

    def _cross_attention(self, kg_embeddings, seq_embeddings):
        """
        Cross-attention between KG embeddings (query) and sequence embeddings (key/value)
        kg_embeddings: [batch_size, dim] - from KG aggregation
        seq_embeddings: [batch_size, max_seq_length, dim] - from sequence processing
        """
        batch_size = tf.shape(kg_embeddings)[0]
        
        # Expand KG embeddings to match sequence dimension
        kg_expanded = tf.expand_dims(kg_embeddings, 1)  # [batch_size, 1, dim]
        
        # Compute Query from KG, Key and Value from sequence
        Q = tf.matmul(tf.reshape(kg_expanded, [-1, self.dim]), self.W_Q)
        Q = tf.reshape(Q, [batch_size, 1, self.dim])
        
        K = tf.matmul(tf.reshape(seq_embeddings, [-1, self.dim]), self.W_K)
        K = tf.reshape(K, [batch_size, self.max_seq_length, self.dim])
        
        V = tf.matmul(tf.reshape(seq_embeddings, [-1, self.dim]), self.W_V)
        V = tf.reshape(V, [batch_size, self.max_seq_length, self.dim])
        
        # Attention scores: Q * K^T / sqrt(d)
        attention_scores = tf.matmul(Q, K, transpose_b=True)  # [batch_size, 1, max_seq_length]
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.dim, tf.float32))
        
        # Mask padding positions
        seq_mask = tf.sequence_mask(self.user_seq_length, self.max_seq_length, dtype=tf.float32)
        seq_mask = tf.expand_dims(seq_mask, 1)  # [batch_size, 1, max_seq_length]
        attention_scores = attention_scores * seq_mask + (1.0 - seq_mask) * (-1e9)
        
        # Softmax attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention to values
        context = tf.matmul(attention_weights, V)  # [batch_size, 1, dim]
        context = tf.squeeze(context, axis=1)  # [batch_size, dim]
        
        return context
    
    def _multi_head_attention(self, inputs):
        """
        Multi-head self-attention - returns full sequence (not pooled)
        inputs: [batch_size, max_seq_length, dim]
        returns: [batch_size, max_seq_length, dim]
        """
        batch_size = tf.shape(inputs)[0]
        head_dim = self.dim // self.num_heads
        
        inputs_flat = tf.reshape(inputs, [-1, self.dim])
        
        Q = tf.matmul(inputs_flat, self.W_Q)
        K = tf.matmul(inputs_flat, self.W_K)
        V = tf.matmul(inputs_flat, self.W_V)
        
        Q = tf.reshape(Q, [batch_size, self.max_seq_length, self.num_heads, head_dim])
        K = tf.reshape(K, [batch_size, self.max_seq_length, self.num_heads, head_dim])
        V = tf.reshape(V, [batch_size, self.max_seq_length, self.num_heads, head_dim])
        
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])
        
        attention_scores = tf.matmul(Q, K, transpose_b=True)
        attention_scores = attention_scores / tf.sqrt(tf.cast(head_dim, tf.float32))
        
        # Mask for padding
        seq_mask = tf.sequence_mask(self.user_seq_length, self.max_seq_length, dtype=tf.float32)
        seq_mask = tf.expand_dims(tf.expand_dims(seq_mask, 1), 1)
        attention_mask = seq_mask * tf.transpose(seq_mask, [0, 1, 3, 2])
        
        attention_scores = attention_scores * attention_mask + (1.0 - attention_mask) * (-1e9)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, V)
        
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, [batch_size, self.max_seq_length, self.dim])
        
        # ✅ ADD THIS: Output projection W_O
        attention_output = tf.matmul(tf.reshape(attention_output, [-1, self.dim]), self.W_O)
        attention_output = tf.reshape(attention_output, [batch_size, self.max_seq_length, self.dim])

        # ✅ Return full sequence, NOT pooled
        return attention_output
        
    def _pool_sequence(self, sequence_output):
        """
        Average pool over valid sequence positions
        sequence_output: [batch_size, max_seq_length, dim]
        returns: [batch_size, dim]
        """
        seq_mask = tf.sequence_mask(self.user_seq_length, self.max_seq_length, dtype=tf.float32)
        seq_mask_expand = tf.expand_dims(seq_mask, -1)
        
        masked_output = sequence_output * seq_mask_expand
        sequence_sum = tf.reduce_sum(masked_output, axis=1)
        
        seq_lengths_float = tf.cast(tf.expand_dims(self.user_seq_length, -1), tf.float32)
        seq_lengths_float = tf.maximum(seq_lengths_float, 1.0)
        
        pooled_output = sequence_sum / seq_lengths_float
        return pooled_output

    def _gate_fusion(self, kg_embedding, seq_embedding):
        """
        Gate fusion mechanism to combine KG and sequence embeddings
        As per document: g = sigmoid(combined), Fused = g⊙KG + (1-g)⊙Seq
        """
        # Combine embeddings
        combined = kg_embedding + seq_embedding
        
        # Compute gate: g = sigmoid(W * combined + b)
        gate = tf.sigmoid(tf.matmul(combined, self.W_gate) + self.b_gate)
        # Shape: [batch_size, dim]
        
        # Fused embedding: g ⊙ KG + (1 - g) ⊙ Sequence
        fused_embedding = gate * kg_embedding + (1.0 - gate) * seq_embedding
        
        return fused_embedding
    def _build_sequential_loss(self):
              
        """
        Sequential next-item prediction loss using negative sampling
        Predicts the next item based on user's historical sequence
        """
        # Get current item embeddings (targets for next-item prediction)
        target_item_emb = tf.nn.embedding_lookup(self.entity_emb_matrix, self.item_indices)
        
        # if self.model_type in ['KGCN_LightGCN', 'LightGCN']:
        #     target_item_emb = tf.nn.embedding_lookup(self.ia_embeddings, self.item_indices)
        # else:
        #     target_item_emb = tf.nn.embedding_lookup(self.entity_emb_matrix, self.item_indices)
        
        # [batch_size, dim]
        
        # Use sequence embeddings as context to predict next item
        # self.user_embeddings_seq contains the aggregated sequence representation
        seq_context = self.user_embeddings_seq
        # [batch_size, dim]
        
        # Positive scores: how well sequence context predicts the actual item
        pos_scores = tf.reduce_sum(seq_context * target_item_emb, axis=1)
        # [batch_size]
        
        # Negative sampling: sample random items as negatives
        batch_size = tf.shape(self.item_indices)[0]
        
        # Sample negative items (same number as batch)
        neg_item_indices = tf.random.uniform(
            shape=[batch_size], 
            minval=0, 
            maxval=self.n_item, 
            dtype=tf.int64
        )
        neg_item_emb = tf.nn.embedding_lookup(self.entity_emb_matrix, neg_item_indices)
        # # ✅ Use consistent embeddings
        # if self.model_type in ['KGCN_LightGCN', 'LightGCN']:
        #     neg_item_emb = tf.nn.embedding_lookup(self.ia_embeddings, neg_item_indices)
        # else:
        #     neg_item_emb = tf.nn.embedding_lookup(self.entity_emb_matrix, neg_item_indices)
        # Negative scores
        neg_scores = tf.reduce_sum(seq_context * neg_item_emb, axis=1)
        # [batch_size]
        
        # BPR loss: maximize difference between positive and negative scores
        # Loss = -log(sigmoid(pos_score - neg_score))
        bpr_loss = -tf.reduce_mean(
            tf.math.log_sigmoid(pos_scores - neg_scores + 1e-10)
        )
        
        # Alternative: You can also use cross-entropy loss
        # Combine pos and neg scores
        # logits = tf.stack([pos_scores, neg_scores], axis=1)
        # labels = tf.one_hot(tf.zeros(batch_size, dtype=tf.int32), depth=2)
        # ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # seq_loss = tf.reduce_mean(ce_loss)
        
        return bpr_loss