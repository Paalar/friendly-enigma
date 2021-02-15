from yaml import safe_load

stream = open("config.yaml", "r")
config = safe_load(stream)
config["device"] = "cpu"