import glob

import wandb
import os
import causal_nf.config as causal_nf_config
import causal_nf.utils.training as causal_nf_train
import causal_nf.utils.wandb_local as wandb_local
from causal_nf.config import cfg
import causal_nf.utils.io as causal_nf_io
from causal_nf.preparators.synthetic_preparator import SyntheticPreparator

os.environ["WANDB_NOTEBOOK_NAME"] = "name_of_the_notebook"

args_list, args = causal_nf_config.parse_args()

load_model = isinstance(args.load_model, str)
if load_model:
    causal_nf_io.print_info(f"Loading model: {args.load_model}")

config = causal_nf_config.build_config(
    config_file=args.config_file,
    args_list=args_list,
    config_default_file=args.config_default_file,
)

causal_nf_config.assert_cfg_and_config(cfg, config)

if cfg.device in ["cpu"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
causal_nf_train.set_reproducibility(cfg)

preparator = SyntheticPreparator.loader(cfg.dataset)
preparator.prepare_data()
loaders = preparator.get_dataloaders(
    batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers
)

for i, loader in enumerate(loaders):
    causal_nf_io.print_info(f"[{i}] num_batchees: {len(loader)}")

model = causal_nf_train.load_model(cfg=cfg, preparator=preparator)

param_count = model.param_count()
config["param_count"] = param_count

if not load_model:
    assert isinstance(args.project, str)
    run = wandb.init(
        mode=args.wandb_mode,
        group=args.wandb_group,
        project=args.project,
        config=config,
    )

# set dirpath
folder = f"{cfg['dataset']['name']}_{cfg['dataset']['nr_nodes']}_{cfg['dataset']['nr_edges']}_{cfg['dataset']['data_seed']}"
dirpath = os.path.join("experiments", "E6", "NF", f"NF_{folder}")
logger_dir = os.path.join("experiments", "E6", "NF", f"NF_{folder}")

trainer, logger = causal_nf_train.load_trainer(
    cfg=cfg,
    dirpath=dirpath,
    logger_dir=logger_dir,
    include_logger=True,
    model_checkpoint=cfg.train.model_checkpoint,
    cfg_early=cfg.early_stopping,
    preparator=preparator,
)

causal_nf_io.print_info(f"Experiment folder: {logger.save_dir}\n\n")

wandb_local.log_config(dict(config), root=logger.save_dir)

if not load_model:
    wandb_local.copy_config(
        config_default=causal_nf_config.DEFAULT_CONFIG_FILE,
        config_experiment=args.config_file,
        root=logger.save_dir,
    )
    import time
    start_time = time.time()
    trainer.fit(model, train_dataloaders=loaders[0], val_dataloaders=loaders[1])
    training_time = time.time() - start_time
    # Save the training time in a file
    training_time_file = os.path.join(logger_dir, "training_time.txt")
    with open(training_time_file, "w") as f:
        f.write(f"Training time: {training_time} seconds\n")


if isinstance(preparator.single_split, str):
    loaders = [loaders[0]]

run.finish()
if args.delete_ckpt:
    for f in glob.iglob(os.path.join(logger.save_dir, "*.ckpt")):
        causal_nf_io.print_warning(f"Deleting {f}")
        os.remove(f)