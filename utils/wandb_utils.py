import wandb
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class WandbLogger:
    def __init__(self, config, output_dir=None, run_name='run', job_type="training"):
        self.config = config
        self.run_name = run_name
        self.output_dir = output_dir
        self.initialized = False
        self.job_type = job_type

    def initialize(self):
        if not self.initialized:
            init_kwargs = {}
            if isinstance(self.config, dict):
                if "project" in self.config:
                    init_kwargs["project"] = self.config["project"]
                if "entity" in self.config:
                    init_kwargs["entity"] = self.config["entity"]
            self.run = wandb.init(
                name=self.run_name,
                config=self.config,
                dir=self.output_dir,
                job_type=self.job_type,
                **init_kwargs,
            )
            self.initialized = True

    def get_config(self):
        return wandb.config

    def log(self, data, step=None):
        if not self.initialized:
            self.initialize()
        wandb.log(data, step=step)

    def log_figures(self, figures, step=None):
        """Log dict of {name: matplotlib.Figure} as wandb.Image. Closes figs after."""
        if not self.initialized:
            self.initialize()
        log_dict = {}
        for name, fig in figures.items():
            log_dict[name] = wandb.Image(fig)
            plt.close(fig)
        wandb.log(log_dict, step=step)

    def save_model(self, model, model_name, optimizer, epoch, output_dir):
        if not self.initialized:
            self.initialize()
        file_path = os.path.join(output_dir, model_name)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, file_path)
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)

    def alert(self, ttl, txt):
        if not self.initialized:
            self.initialize()
        self.run.alert(title=ttl, text=txt)

    def finish(self):
        if self.initialized:
            wandb.finish()
            self.initialized = False
