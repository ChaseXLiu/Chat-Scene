import torch
data = torch.load("annotations/scannet_mask3d_videofeats_dinov2.pt", map_location="cpu")

print(type(data))
if isinstance(data, dict):
    print("Keys:", list(data.keys())[:5])
    first_key = list(data.keys())[0]
    print("Type of first value:", type(data[first_key]))
    if isinstance(data[first_key], dict):
        print("Inner keys:", data[first_key].keys())
        for k, v in data[first_key].items():
            if isinstance(v, torch.Tensor):
                print(f" - {k}: {v.shape}")
elif isinstance(data, list):
    print("List length:", len(data))
    print("Type of first element:", type(data[0]))
elif isinstance(data, torch.Tensor):
    print("Tensor shape:", data.shape)
