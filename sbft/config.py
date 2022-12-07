import typing as t
from datetime import datetime

import pydantic as pdt
import torch


class WandBConfig(pdt.BaseModel):
    """WandB configuration"""

    project_name: str = pdt.Field(default="SBFT", description="WandB project name.")
    entity: str = pdt.Field(default="SBFT", description="The entity (team) name.")
    capture_video: bool = pdt.Field(
        default=False, description="If True, capture videos of the agent performance."
    )

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    @pdt.validator("capture_video")
    def check_capture_video(cls, value: bool):
        if value:
            raise NotImplementedError(
                "Video capture support has not yet been implemented in SBFT"
            )

    @property
    def init_config(self) -> dict[str, t.Any]:
        return dict(
            project=self.project_name,
            entity=self.entity,
            save_code=True,
        )


class ExperimentConfig(pdt.BaseModel):
    """The base class for experiment configuration in sbft.

    The sbft cli knows how to load experiment configuration in a given file.
    """

    exp_name: str = pdt.Field(
        default="SBFT_Experiment", description="Name of experiment."
    )
    seed: int = pdt.Field(default=0, description="Seed for reproducibility.")
    cuda_deterministic: bool = pdt.Field(
        default=True,
        description="If toggled, sets `torch.cuda.manual_seed_all(seed)`.",
    )
    cudnn_deterministic: bool = pdt.Field(
        default=True,
        description="If toggled, sets `torch.backends.cudnn.deterministic=False`.",
    )
    cuda: bool = pdt.Field(default=False, description="Whether to use CUDA.")
    wandb: bool = pdt.Field(
        default=False, description="Whether to track the experiment in WandB."
    )
    wandb_config: WandBConfig = pdt.Field(
        default_factory=WandBConfig,
        description="Config for WandB tracking.",
        exclude=True,
    )

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    @property
    def torch_device(self) -> torch.device:
        if self.cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def wandb_init_config(self) -> dict[str, t.Any]:
        result = self.wandb_config.init_config
        # Time formatted as Mon_Jan_06_22__15_32_55
        formatted_time = datetime.strftime(datetime.now(), "%a_%b_%d_%y__%H_%M_%S")
        result["name"] = f"{self.exp_name}__{self.seed}__{formatted_time}"
        result["config"] = self.dict()
        return result
