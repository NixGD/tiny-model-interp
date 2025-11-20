# train a character-level model on Gutenberg-BookCorpus dataset
# uses a small subset of the full dataset

out_dir = "models/web-char-11-20"
eval_interval = 500
eval_iters = 25
log_interval = 50

# save checkpoints when validation improves
always_save_checkpoint = True

wandb_log = True  # override via command line if you like
wandb_project = "tiny-model-training"
wandb_run_name = "test-rezero"

dataset = "fineweb_char"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

learning_rate = 1e-5  # with baby networks can afford to go a bit higher
max_iters = 10_000
lr_decay_iters = 10_000  # make equal to max_iters usually
min_lr = 5e-3  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
# device = "cpu"  # run on cpu only
# compile = False  # do not torch compile the model
