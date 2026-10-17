import sys, json, torch, importlib.util
out = {}
try:
    import torch_scatter as real
    spec = importlib.util.spec_from_file_location('shim', sys.argv[1] + '/torch_scatter/__init__.py'); shim = importlib.util.module_from_spec(spec); spec.loader.exec_module(shim)
    g = torch.Generator(device='cuda').manual_seed(0); md = 0.0
    for dt in [torch.float32, torch.float16]:
        for _ in range(5):
            n = 50000; src = torch.randn(n, 64, device='cuda', generator=g).to(dt); idx = torch.randint(0, 7000, (n,), device='cuda', generator=g)
            a = real.scatter(src, index=idx, dim=0, dim_size=7000, reduce='sum').float(); b = shim.scatter(src, index=idx, dim=0, dim_size=7000, reduce='sum').float()
            md = max(md, float((a - b).abs().max() / (a.abs().max() + 1e-9)))
    out = {'real': real.__version__, 'max_rel_diff': md}
except Exception as e:
    out = {'error': repr(e)[:300]}
print(json.dumps(out))
