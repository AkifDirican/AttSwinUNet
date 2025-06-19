import sys
import types
import builtins
import numpy as np
import torch
import pytest
from .inference import inference
import io
import contextlib
import argparse
import importlib
import test

# test_test.py


# Patch absolute import for inference

def test_inference_called(monkeypatch):
    # Mock Net and test_loader
    class DummyNet(torch.nn.Module):
        def eval(self): pass
        def forward(self, x): return torch.ones((1,1,2,2))
        def __call__(self, x): return self.forward(x)
    class DummyLoader:
        def __iter__(self): return iter([{'image': torch.ones((1,3,2,2)), 'mask': torch.ones((1,1,2,2))}])
        def __len__(self): return 1

    net = DummyNet()
    loader = DummyLoader()

    # Capture print output
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        inference(net, loader)
    output = f.getvalue()
    assert "F1 score" in output
    assert "Accuracy" in output
    assert "Specificity" in output
    assert "Sensitivity" in output

def test_script_runs(monkeypatch):
    # Patch all dependencies in test.py
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.load", lambda *a, **kw: {'model_weights': {}})
    monkeypatch.setattr("torch.nn.Module.load_state_dict", lambda self, x: None)
    monkeypatch.setattr("torch.nn.Module.to", lambda self, device: self)
    monkeypatch.setattr("torch.nn.Module.cuda", lambda self: self)
    monkeypatch.setattr("yaml.load", lambda *a, **kw: {'number_classes': 1, 'path_to_data': 'dummy'})
    monkeypatch.setattr("builtins.open", lambda *a, **kw: None)
    # Dummy config
    dummy_config = types.SimpleNamespace(get_swin_unet_attention_configs=lambda: types.SimpleNamespace(to_dict=lambda: {'volume_path': 'dummy', 'num_classes': 1}))
    sys.modules['configs.swin_attention_unet'] = dummy_config
    # Dummy DataLoader and dataset
    class DummyDataset:
        def __init__(self, *a, **kw): pass
        def __len__(self): return 1
        def __getitem__(self, idx): return {'image': torch.ones((1,3,2,2)), 'mask': torch.ones((1,1,2,2))}
    class DummyLoader:
        def __init__(self, *a, **kw): pass
        def __iter__(self): return iter([{'image': torch.ones((1,3,2,2)), 'mask': torch.ones((1,1,2,2))}])
        def __len__(self): return 1
    sys.modules['loader'] = types.SimpleNamespace(isic_loader=DummyDataset)
    sys.modules['torch.utils.data'] = types.SimpleNamespace(DataLoader=DummyLoader)
    # Dummy model
    class DummyNet(torch.nn.Module):
        def __init__(self, *a, **kw): pass
        def eval(self): pass
        def forward(self, x): return torch.ones((1,1,2,2))
        def __call__(self, x): return self.forward(x)
        def load_state_dict(self, x): pass
        def to(self, device): return self
        def cuda(self): return self
    sys.modules['model.attention_swin_unet'] = types.SimpleNamespace(SwinAttentionUnet=DummyNet)
    # Patch argparse
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self, args=None: types.SimpleNamespace(num_classes=1, volume_path='dummy'))
    # Patch inference to check call
    called = {}
    def fake_inference(Net, test_loader):
        called['called'] = True
    monkeypatch.setattr("test.inference", fake_inference)
    # Import test.py as a module (simulate running)
    assert called.get('called', False)