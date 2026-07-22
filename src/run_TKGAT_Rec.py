
import subprocess
import datetime
import os

# Hyperparameter search space
l2_weight_values = ["1e-6"]
lr_values = [2e-2]
max_seq_len_values = [20]
num_heads_values = [2]
seq_weight_values = [0.7]
alpha_values = [0.0]

for l2_weight in l2_weight_values:
    for lr in lr_values:
        for max_seq_len in max_seq_len_values:
            for num_heads in num_heads_values:
                for seq_weight in seq_weight_values:
                    for alpha in alpha_values:
                        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        log = (
                            f"withoutTemoporal(GCN + KG)_Bi-Product_Movie_l2{l2_weight}_lr{lr}_seq{max_seq_len}"
                            f"h{num_heads}_sw{seq_weight}_alpha{alpha}_{ts}.txt"
                        )

                        print(
                            f">>> Running l2={l2_weight} | lr={lr} | seq_len={max_seq_len} "
                            f"| heads={num_heads} | seq_weight={seq_weight} | alpha={alpha} | log: {log}"
                        )

                        cmd = [
                            "python", "main.py",
                            "--dataset", "movie",
                            "--aggregator", "concat",
                            "--n_epochs", "15",
                            "--neighbor_sample_size", "4",
                            "--dim", "32",
                            "--n_iter", "2",
                            "--batch_size", "65536",
                            "--l2_weight", str(l2_weight),
                            "--lr", str(lr),
                            "--layer_size", "[32]",
                            "--adj_type", "plain",
                            "--alg_type", "lightgcn",
                            "--model_type", "KGCN_LightGCN",
                            # "--model_type", "KGCN",  # ← pure KGCN (no CF)
                            # "--alg_type", "ngcf",    # ← ignored, but keeps argparse happy
                            "--node_dropout", "[0.1]",
                            "--mess_dropout", "[0.1]",
                            "--node_dropout_flag", "1",
                            "--alpha", str(alpha),
                            "--smoothing_steps", "1",
                            "--pretrain", "0",
                            "--att", "h_ur_t",
                            "--runs", "3",
                            "--gpu_id", "0",
                            "--use_temporal", "0",
                            "--max_seq_length", str(max_seq_len),
                            "--num_heads", str(num_heads),
                            "--seq_weight", str(seq_weight),
                        ]

                        with open(log, "w") as f:
                            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                            process.wait()


# --att h_ur_t → Product Attention
# --att uhrt_concat → Concat Attention
# --att u_h_r_t_mlp → Bi_Perceptron Attention
# --att uhrt_bi → Bi_Interaction Attention
