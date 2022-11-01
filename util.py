import yaml

config = None
#获取配置文件
def getconfig(config_path):
    if config is not None:
        return config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config