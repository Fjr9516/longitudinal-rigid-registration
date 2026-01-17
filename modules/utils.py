import yaml

def load_config(config_path):
	"""Load a YAML configuration file from the given path."""
	with open(config_path, 'r') as f:
		return yaml.safe_load(f)
