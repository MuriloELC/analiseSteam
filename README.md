Para gerar a amostra (uma vez):
python games_analyzer.py --csv steam_games.csv --generate-sample

Para preencher e validar sua amostra:
python games_analyzer.py --csv steam_games.csv --run-sample-tests

Para ver as respostas no console:
python games_analyzer.py --csv steam_games.csv

Para gerar o relatório:
python games_analyzer.py --csv steam_games.csv --report report.md

Para suprimir saídas, ver detalhes das linhas inválidas e limitar gêneros (ex.: top 10):
python games_analyzer.py --csv steam_games.csv --quiet --debug-invalid --top-n 10

