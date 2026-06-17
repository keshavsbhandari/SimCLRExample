from omegaconf import OmegaConf

from simclr_fraud.train.utils import resolve_wandb_project


def test_resolve_wandb_project_prefers_config(monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "from-env")
    cfg = OmegaConf.create({"wandb": {"project": "from-yaml"}})
    project, source = resolve_wandb_project(cfg)
    assert project == "from-yaml"
    assert source == "config"


def test_resolve_wandb_project_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("WANDB_PROJECT", "from-env")
    cfg = OmegaConf.create({"wandb": {"project": ""}})
    project, source = resolve_wandb_project(cfg)
    assert project == "from-env"
    assert source == "WANDB_PROJECT env"


def test_resolve_wandb_project_raises_when_unset(monkeypatch):
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    cfg = OmegaConf.create({"wandb": {}})
    try:
        resolve_wandb_project(cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
