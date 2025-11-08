import os
import json

def get_config():
    if os.path.isfile("config.json"):
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_config(cfg):
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

def missing_values():
    cfg = {}
    cfg["api_key"] = input("api key da speciesLink: ").strip().strip('"')
    cfg["db_user"] = input("usuário myswl: ").strip().strip('"')
    cfg["db_password"] = input("senha mysql: ").strip().strip('"')
    cfg["db_host"] = input("host mysql (ex: 127.0.0.1): ").strip().strip('"')
    cfg["db_schema"] = input("nome do schema: ").strip().strip('"')
    cfg["db_table"] = input("tabela de destino mysql: ").strip().strip('"')
    save_config(cfg)
    return cfg
