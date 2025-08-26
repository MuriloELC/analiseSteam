# Games Analyzer — Relatório

## Metodologia
- **Parsing de dados**: datas em múltiplos formatos; fallback por regex de ano (`\b(19|20)\d{2}\b`).
- **Preço**: limpeza de símbolos/textos; vírgulas → ponto; palavras como *Free/Gratuito* → 0.0.
- **Gratuito**: inferido por `Price == 0.0` (quando `price` válido). Preços inválidos contam como **não grátis**.
- **Gêneros**: separados por vírgulas, com espaços aparados; vazios descartados.
- **Linhas inválidas**: processamento *fail-soft* (apenas casos essenciais, como `Name` vazio).

## Pergunta 1 — % grátis vs pagos

| Métrica | Valor |
|---|---|
| Jogos grátis | 12679 |
| Jogos pagos | 60253 |
| Total | 72932 |
| % grátis | 17.4% |
| % pagos | 82.6% |

**Discussão**: a razão grátis/pago sugere estratégia de monetização (F2P, DLCs, cosméticos),
além de efeitos de catálogo legado e promoções.

## Pergunta 2 — Anos com mais lançamentos

- **Pico**: 2022 (máximo = 13961)

**Reflexão**: picos podem refletir fatores externos (geração de consoles, motores acessíveis,
plataformas de distribuição digital, choques macroeconômicos).

## Pergunta 3 — Gêneros nos anos de pico

- Anos de pico: 2022

| Rank | Gênero | Contagem |
|---:|---|---:|
| 1 | Indie | 8712 |
| 2 | Casual | 5683 |
| 3 | Adventure | 5102 |
| 4 | Action | 5101 |
| 5 | Simulation | 2558 |

**Por quê isso importa?** Concentrar gêneros de maior tração nos anos de maior oferta ajuda
decisões de portfólio, marketing e posicionamento competitivo.
