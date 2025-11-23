# ciencia-de-dados
Código do trabalho final da disciplina de Ciência de Dados, UTFPR-CM.

Baseado no [speciesLink:](https://specieslink.net/)

IMPORTANTE: para rodar o código você precisa ter uma api key, que você pode obter ao fazer uma conta (grátis) no speciesLink! Professor, caso o senhor não queira passar pelo processo de pegar os registros na API, recomendo usar o arquivo /etapa_2/registros_biodiversidade.csv, que foi gerado após as filtragens em filtragem.ipynb.

## Como usar:
- instale todos os requerimentos do requirements.txt;
- tenha o MySQL instalado - a integração com a API do speciesLink coloca os dados em uma tabela pré-criada do mysql!
  você pode usar a tabela ciencia_de_dados.sql, também nesse repositório. Apenas altere o nome da schema.

- Em etapa_2:
  - python main.py buscar --family Piperaceae (foi a família que utilizamos) --output nome do arquivo.json 
  - python main.py inserir --input nome do arquivo.json
  - realizar as filtragens encontradas em filtragem.ipynb

- Em etapa_3:
  - 3.2.1 Organização dos Dados
  - 3.2.2 Reestruturação Necessária
  - 3.3 Consultas SQL Analíticas
  - 3.4 Análise Exploratória e Teste de Hipóteses

- Em etapa_4:
  - teste.py tem o 4.2 da primeira pergunta e forest.ipynb tem o 4.2 da segunda pergunta;