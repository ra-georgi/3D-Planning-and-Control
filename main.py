import yaml

with open("config/default.yaml", "r") as f:
    sim_cfg = yaml.safe_load(f)

# print(cfg["vehicle"]["mass"])

