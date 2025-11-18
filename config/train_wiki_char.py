# train a character-level model on Gutenberg-BookCorpus dataset
# uses a small subset of the full dataset

out_dir = "out-wiki-char"
eval_interval = 250
eval_iters = 25
log_interval = 25

# save checkpoints when validation improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = "gutenberg-char"
wandb_run_name = "mini-gpt"

dataset = "wiki_char"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

learning_rate = 7e-4  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
device = 'cpu'  # run on cpu only
compile = False  # do not torch compile the model
