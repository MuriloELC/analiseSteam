Para gerar a amostra (uma vez):
python games_analyzer.py --csv steam_games.csv --generate-sample

Para preencher e validar sua amostra:
python games_analyzer.py --csv steam_games.csv --run-sample-tests

Para ver as respostas no console:
python games_analyzer.py --csv steam_games.csv

Para gerar o relatório:
python games_analyzer.py --csv steam_games.csv --report report.md

Para ver detalhes das linhas realmente inválidas (ex.: Name vazio):
python games_analyzer.py --csv steam_games.csv --debug-invalid