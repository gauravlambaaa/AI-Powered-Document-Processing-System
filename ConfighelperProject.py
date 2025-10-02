import yaml

with open('AppConfig/config.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

def get_config(key):
    value = data.get(key)
    return value
