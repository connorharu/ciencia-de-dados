import requests
import pymysql
import pandas as pd
from pandas import json_normalize

class SpeciesLink:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://specieslink.net/ws/1.0/search"

    def search_records(self, filters: dict): # pegar os registros que se enquadram na busca na API
        params = {"apikey": self.api_key}
        params.update(filters)

        all_records = {"features": []}
        offset = 0
        limit = 5000 # tem um limite por busca, como se fosse a "paginação" da API, a idéia é ir incrementando o offset pra "passar a página" e ler mais dados
        total_found = None

        while True:
            params.update({"offset": offset, "limit": limit})
            response = requests.get(self.base_url, params=params)

            if response.status_code != 200:
                print("erro:", response.status_code)
                break

            payload = response.json()
            if total_found is None:
                total_found = payload.get("numberMatched", 0)

            batch = payload.get("features", [])
            all_records["features"].extend(batch)

            print(f"até agora: {len(all_records['features'])} / {total_found}")
            if len(all_records["features"]) >= total_found:
                break

            offset += limit # virar pagina

        print("busca concluída com sucesso.")
        return all_records

    def insert_into_mysql(self, records: dict, db_config: dict, table_name: str):
        try:
            connection = pymysql.connect(**db_config)

            if connection.open: # se a conexão está ativa
                print("conexão estabelecida")
            else:
                return
            
            cursor = connection.cursor()

            df = json_normalize(records, "features", errors="ignore")
            df = df.where(pd.notnull(df), None)

            for idx, row in df.iterrows():
                print(f"inserindo registro {idx+1}...")

                data = row.to_dict()
                columns = [
                    "barcode", "collectioncode", "catalognumber", "scientificname", "kingdom",
                    "family", "genus", "yearcollected", "monthcollected", "daycollected", "country",
                    "stateprovince", "county", "locality", "institutioncode", "phylum", "basisofrecord",
                    "verbatimlatitude", "verbatimlongitude", "identifiedby", "collectionid",
                    "specificepithet", "recordedby", "decimallongitude", "decimallatitude",
                    "modified", "scientificnameauthorship", "recordnumber", "occurrenceremarks"
                ]

                values = tuple(data.get(f"properties.{c}", None) for c in columns)

                placeholders = ", ".join(["%s"] * len(columns))
                insert_sql = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES ({placeholders})
                """

                cursor.execute(insert_sql, values)
                print(f"eegistro {idx + 1} inserido")

            connection.commit()
            print(f"fim - {len(df)} registros adicionados")

        except pymysql.MySQLError as err:
            print("erro no mysql:", err)

        except Exception as e:
            print("erro inesperado:", e)

        finally:
            if connection and connection.open:
                connection.close()
                print("conexão encerrada")
