import argparse
import json
from speciesLink import SpeciesLink
from config import get_config, missing_values

def main():
    config = get_config()
    if config is None or any(k not in config for k in ['api_key', 'db_user', 'db_password', 'db_host', 'db_schema', 'db_table']):
        config = missing_values()

    parser = argparse.ArgumentParser(description="ferramenta de integração com a speciesLink")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # buscar registros na api do specieslink
    buscar = subparsers.add_parser("buscar", help="buscar registros na API speciesLink")
    buscar.add_argument("--family", help="filtrar por família")
    buscar.add_argument("--output", default="resultados.json", help="arquivo de saida JSON")

    # inserir registros no banco de dados - há um esqueleto de banco de dados que pode ser utilizado nesse repositório!
    inserir = subparsers.add_parser("inserir", help="inserir registros no banco de dados")
    inserir.add_argument("--input", required=True, help="arquivo JSON com registros a inserir")

    args = parser.parse_args()
    specieslink = SpeciesLink(config["api_key"])

    if args.command == "buscar":
        args_dict = vars(args)  # transforma argumentos em dicionário
        filters = {}
        for key, value in args_dict.items():
            if value and key not in ["command", "output"]: # ignora o comando buscar e o output do json, pegando só os argumentos da filtragem
                filters[key] = value
        results = specieslink.search_records(filters)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"resultados salvos em {args.output}")

    elif args.command == "inserir":
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        db_config = {
            "host": config["db_host"],
            "user": config["db_user"],
            "password": config["db_password"],
            "database": config["db_schema"],
        }
        specieslink.insert_into_mysql(data, db_config, config["db_table"])

if __name__ == "__main__":
    main()
