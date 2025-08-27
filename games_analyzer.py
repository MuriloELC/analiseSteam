#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
games_analyzer.py
-----------------
Analisador de jogos a partir de um CSV, usando apenas a biblioteca padrão.

Recursos:
- Carrega dataset CSV e faz parsing robusto (datas, preços, listas de gêneros).
- API OO para consultas (percentual grátis/pago, anos de pico, gêneros no(s) ano(s) de pico).
- Gera amostra determinística (20 jogos) excluindo os 20 primeiros registros.
- Valida resultados sobre a amostra contra expected_results.json (gera template se ausente).
- Funciona por CLI e por import.
- Gera relatório Markdown (report.md).

Dependências: apenas stdlib (csv, argparse, pathlib, dataclasses, datetime, re, json,
random, textwrap, itertools, collections, io, typing, unittest, doctest, statistics opcional).

Exemplos:
  Gerar amostra e relatório:
    python games_analyzer.py --csv data/games.csv --generate-sample --report report.md

  Preparar template e rodar testes da amostra:
    python games_analyzer.py --csv data/games.csv --run-sample-tests

  Rodar as respostas no console:
    python games_analyzer.py --csv data/games.csv
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Iterable

import argparse
import csv
import doctest
import io
import json
import random
import re
import sys
import textwrap
import unittest
from collections import Counter


# ======================================================================================
# Exceções
# ======================================================================================


class DataValidationError(Exception):
    """Erro de dados/parse.

    Usado para sinalizar problemas estruturais (ex.: CSV sem colunas obrigatórias)
    ou operações impossíveis (ex.: dataset curto para amostragem).
    """


# ======================================================================================
# Modelo
# ======================================================================================


@dataclass(frozen=True)
class Game:
    """Entidade de jogo.

    Atributos:
      name: nome do jogo (str | None)
      is_free: derivado de price == 0.0 quando price válido; se price inválido → False
      price: float | None
      year: int | None
      genres: list[str]
      publisher: str | None
      developer: str | None
      raw: linha original normalizada (apenas colunas relevantes)

    __post_init__ faz validações e normalizações básicas.
    """

    name: Optional[str]
    is_free: bool
    price: Optional[float]
    year: Optional[int]
    genres: list[str]
    publisher: Optional[str]
    developer: Optional[str]
    raw: dict[str, str]

    def __post_init__(self) -> None:
        # Tipos básicos
        if self.name is not None:
            if not isinstance(self.name, str) or not self.name.strip():
                raise DataValidationError("Game.name inválido (string vazia ou não-string).")
        if self.price is not None and not isinstance(self.price, float):
            raise DataValidationError("Game.price deve ser float ou None.")
        if self.year is not None and not isinstance(self.year, int):
            raise DataValidationError("Game.year deve ser int ou None.")
        if not isinstance(self.genres, list):
            raise DataValidationError("Game.genres deve ser list[str].")
        for g in self.genres:
            if not isinstance(g, str):
                raise DataValidationError("Game.genres contém elemento não-string.")
        # Normalizações (aceita lista vazia de gêneros)
        object.__setattr__(self, "genres", [g.strip() for g in self.genres if g.strip()])
        object.__setattr__(self, "publisher", self.publisher.strip() if self._nz(self.publisher) else None)
        object.__setattr__(self, "developer", self.developer.strip() if self._nz(self.developer) else None)

        # raw deve conter as chaves normalizadas
        expected_keys = {"Name", "Release date", "Price", "Genres", "Developers", "Publishers"}
        missing = expected_keys - set(self.raw.keys())
        if missing:
            raise DataValidationError(f"Game.raw faltando chaves: {sorted(missing)}")

    @staticmethod
    def _nz(val: Optional[str]) -> bool:
        return isinstance(val, str) and val.strip() != ""


# ======================================================================================
# Parsing
# ======================================================================================


class GameParser:
    """Conversor de dict[str,str] → Game com tratamento robusto.

    Doctests rápidos:

    >>> GameParser.parse_price("0.00")
    0.0
    >>> GameParser.parse_year("Oct 21, 2008")
    2008
    >>> GameParser.parse_genres("Action, Indie, RPG")
    ['Action', 'Indie', 'RPG']
    """

    # Tentativas de formatação de datas comuns
    DATE_FORMATS = [
        "%b %d, %Y",  # Oct 21, 2008
        "%B %d, %Y",  # October 21, 2008
        "%b %Y",      # Oct 2008
        "%B %Y",      # October 2008
        "%Y-%m-%d",   # 2008-10-21
        "%Y/%m/%d",   # 2008/10/21
        "%d %b %Y",   # 21 Oct 2008
        "%d %B %Y",   # 21 October 2008
        "%Y",         # 2008
    ]
    YEAR_REGEX = re.compile(r"\b(19|20)\d{2}\b")

    @staticmethod
    def parse_price(text: Optional[str]) -> Optional[float]:
        """Converte string de preço para float.

        Regras:
          - troca vírgula por ponto se houver
          - remove símbolos e textos extras, mantendo o primeiro número encontrado
          - aceita palavras como 'free' → 0.0
          - se falhar: retorna None

        >>> GameParser.parse_price("$19.99")
        19.99
        >>> GameParser.parse_price("19,99 USD")
        19.99
        >>> GameParser.parse_price("Free")
        0.0
        >>> GameParser.parse_price("")
        >>> GameParser.parse_price(None)
        """
        if text is None:
            return None
        s = str(text).strip()
        if not s:
            return None
        if s.lower() in {"free", "gratuito", "gratis"}:
            return 0.0
        # localizar primeiro número
        match = re.search(r"[-+]?\d+(?:[.,]\d+)?", s)
        if not match:
            return None
        num = match.group(0).replace(",", ".")
        try:
            return float(num)
        except ValueError:
            return None

    @staticmethod
    def parse_year(date_text: Optional[str]) -> Optional[int]:
        """Extrai ano de várias formatações possíveis.

        Tenta vários formatos com strptime; se falhar, tenta regex de ano.
        Se nada funcionar: retorna None.

        >>> GameParser.parse_year("Oct 21, 2008")
        2008
        >>> GameParser.parse_year("2009")
        2009
        >>> GameParser.parse_year("Released 2012 on PC")
        2012
        >>> GameParser.parse_year("")
        >>> GameParser.parse_year(None)
        """
        if date_text is None:
            return None
        s = str(date_text).strip()
        if not s:
            return None
        for fmt in GameParser.DATE_FORMATS:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.year
            except Exception:
                pass
        # regex fallback
        m = GameParser.YEAR_REGEX.search(s)
        if m:
            return int(m.group(0))
        return None

    @staticmethod
    def parse_genres(text: Optional[str]) -> list[str]:
        """Divide string de gêneros por vírgula, removendo vazios.

        >>> GameParser.parse_genres("Action, Indie, RPG")
        ['Action', 'Indie', 'RPG']
        >>> GameParser.parse_genres(" ,  ,  ")
        []
        >>> GameParser.parse_genres(None)
        []
        """
        if text is None:
            return []
        parts = [p.strip() for p in str(text).split(",")]
        return [p for p in parts if p]

    @staticmethod
    def parse(row: dict[str, str]) -> Game:
        """Converte uma linha normalizada (ver GameDataset) em Game.

        is_free = (price == 0.0) quando price válido; caso inválido → price=None e is_free=False.
        Se "Name" estiver vazio, o atributo ``name`` será ``None``.

        >>> r = {'Name':'Foo', 'Release date':'Oct 21, 2008', 'Price':'0.00', 'Genres':'Action, Indie', 'Developers':'ACME', 'Publishers':'ACME'}
        >>> g = GameParser.parse(r)
        >>> g.is_free
        True
        >>> g.year
        2008
        >>> g.genres
        ['Action', 'Indie']
        """
        # Normaliza valores a strings (ou "") via compreensão
        fields = {
            k: str(row.get(k, "") or "").strip()
            for k in ("Name", "Release date", "Price", "Genres", "Developers", "Publishers")
        }

        name = fields["Name"] or None
        rel = fields["Release date"]
        price_raw = fields["Price"]
        genres_raw = fields["Genres"]
        dev = fields["Developers"]
        pub = fields["Publishers"]

        price = GameParser.parse_price(price_raw)
        year = GameParser.parse_year(rel)
        genres = GameParser.parse_genres(genres_raw)  # pode ser []

        is_free = (price == 0.0) if (price is not None) else False

        raw_norm = fields.copy()
        raw_norm["Name"] = name or ""

        return Game(
            name=name,
            is_free=is_free,
            price=price,
            year=year,
            genres=genres,
            publisher=pub or None,
            developer=dev or None,
            raw=raw_norm,
        )


# ======================================================================================
# Dataset
# ======================================================================================


class GameDataset:
    """Carregador e utilitários de dataset."""

    REQUIRED = ["Name", "Release date", "Price", "Genres"]
    OPTIONAL = ["Developers", "Publishers"]

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)
        self.invalid_rows_count: int = 0
        self.invalid_rows_detail: list[tuple[int, str]] = []
        self.loaded_field_map: dict[str, str] = {}  # canonical -> original field name

    @staticmethod
    def _canon(s: str) -> str:
        return s.strip().lower().replace("_", " ")

    def _build_field_map(self, fieldnames: list[str]) -> dict[str, str]:
        if not fieldnames:
            raise DataValidationError("CSV sem cabeçalho (fieldnames vazias).")
        inv = {self._canon(f): f for f in fieldnames}
        field_map: dict[str, str] = {}
        # Mapeia obrigatórias
        for key in self.REQUIRED + self.OPTIONAL:
            c = self._canon(key)
            if c in inv:
                field_map[key] = inv[c]
            else:
                if key in self.REQUIRED:
                    raise DataValidationError(
                        f"CSV sem coluna obrigatória: '{key}'. "
                        f"Encontradas: {fieldnames}"
                    )
                else:
                    field_map[key] = key  # ausente; será tratada como vazia
        return field_map

    def _normalize_row(self, row: dict[str, Any], field_map: dict[str, str]) -> dict[str, str]:
        # Retorna apenas colunas relevantes com chaves normalizadas
        out: dict[str, str] = {}
        for key in self.REQUIRED + self.OPTIONAL:
            orig = field_map.get(key, key)
            val = row.get(orig, "")
            out[key] = "" if val is None else str(val)
        return out

    def _read_games(self, csv_path: Path, record_invalid: bool) -> list[Game]:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise DataValidationError(f"Arquivo CSV não encontrado: {csv_path}")
        games: list[Game] = []
        if record_invalid:
            self.invalid_rows_count = 0
            self.invalid_rows_detail = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            field_map = self._build_field_map(fieldnames)
            if record_invalid:
                self.loaded_field_map = field_map
            for idx, row in enumerate(reader, start=2):
                try:
                    norm = self._normalize_row(row, field_map)
                    game = GameParser.parse(norm)
                    games.append(game)
                except DataValidationError as e:
                    if record_invalid:
                        self.invalid_rows_count += 1
                        self.invalid_rows_detail.append((idx, str(e)))
                    continue
        return games

    def load(self) -> list[Game]:
        """Lê o CSV principal e retorna lista de Game (fail-soft).

        Agora aceita `Genres` vazio (vira lista []), e só marca inválido quando
        há um problema essencial (ex.: `Name` vazio).
        """
        return self._read_games(self.csv_path, record_invalid=True)

    def load_from_file(self, csv_file: Path) -> list[Game]:
        """Carrega jogos de qualquer CSV (mesma lógica de parsing/normalização)."""
        return self._read_games(Path(csv_file), record_invalid=False)

    def sample_once(self, output_dir: Path, k: int = 20, seed: int = 1234) -> Path:
        """Cria sample/sample.csv determinística (não sobrescreve se existir).

        Regras:
          - Excluir os 20 primeiros registros do dataset completo.
          - Exigir pelo menos k+20 registros.

        Retorna o caminho do arquivo de amostra.

        Lança DataValidationError se insuficiente.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "sample.csv"
        if out_file.exists():
            return out_file

        games = self.load()
        if len(games) < k + 20:
            raise DataValidationError(
                f"Dataset com {len(games)} registros; precisa de pelo menos {k+20} para amostrar."
            )

        rng = random.Random(seed)
        # Somente índices a partir do 20 (exclui 0..19)
        candidates = list(range(20, len(games)))
        chosen_idx = rng.sample(candidates, k)
        chosen = [games[i] for i in chosen_idx]

        # Escreve CSV com as colunas normalizadas
        fieldnames = self.REQUIRED + self.OPTIONAL
        with out_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for g in chosen:
                writer.writerow(g.raw)
        return out_file


# ======================================================================================
# Consultas (Analytics)
# ======================================================================================


class GameAnalytics:
    """Consultas de agregação."""

    def __init__(self, games: list[Game]) -> None:
        self.games = list(games)

    def pct_free_vs_paid(self) -> dict[str, Any]:
        """Percentuais de jogos grátis vs pagos.

        Retorna:
          {"free_pct": float, "paid_pct": float,
           "counts": {"free": int, "paid": int, "total": int}}

        Percentuais em 0–100, com 1 casa decimal.
        """
        total = len(self.games)
        free = sum(1 for g in self.games if g.is_free)
        paid = total - free
        if total == 0:
            free_pct = paid_pct = 0.0
        else:
            free_pct = round(100.0 * free / total, 1)
            paid_pct = round(100.0 * paid / total, 1)
        return {
            "free_pct": free_pct,
            "paid_pct": paid_pct,
            "counts": {"free": free, "paid": paid, "total": total},
        }

    def years_with_most_new_games(self) -> dict[str, Any]:
        """Conta jogos por ano e retorna anos de pico e max_count.

        Ignora year=None.

        Exemplo (doctest simples):

        >>> g1 = Game("A", True, 0.0, 2020, [], None, None, _mkraw("A","2020","0",""))
        >>> g2 = Game("B", False, 10.0, 2021, [], None, None, _mkraw("B","2021","10",""))
        >>> g3 = Game("C", False, 9.0, 2021, [], None, None, _mkraw("C","2021","9",""))
        >>> res = GameAnalytics([g1,g2,g3]).years_with_most_new_games()
        >>> res["max_count"], res["years"]
        (2, [2021])
        """
        counter: Counter[int] = Counter()
        for g in self.games:
            if isinstance(g.year, int):
                counter[g.year] += 1
        if not counter:
            return {"max_count": 0, "years": []}
        max_count = max(counter.values())
        years = sorted([y for y, c in counter.items() if c == max_count])
        return {"max_count": max_count, "years": years}

    def top_genres_in_peak_year(self, top_n: int = 5) -> dict[str, Any]:
        """Gêneros mais frequentes nos anos de pico.

        Passos:
          1) Usa years_with_most_new_games para obter ano(s) de pico.
          2) Filtra jogos desses anos.
          3) Agrega por gênero (varrendo listas).
          4) Ordena por frequência desc; desempate alfabético.
          5) Retorna apenas top_n.

        Se não houver gêneros válidos, retorna estrutura coerente vazia.
        """
        yinfo = self.years_with_most_new_games()
        peak_years: list[int] = yinfo.get("years", [])  # type: ignore
        if not peak_years:
            return {"peak_years": [], "genres_ranked": [], "top_n": int(top_n)}

        counter: Counter[str] = Counter()
        peak_set = set(peak_years)
        for g in self.games:
            if g.year in peak_set and g.genres:
                for gen in g.genres:
                    counter[gen] += 1
        if not counter:
            return {"peak_years": peak_years, "genres_ranked": [], "top_n": int(top_n)}

        items = list(counter.items())
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        ranked = [{"genre": k, "count": v} for k, v in items[: max(0, int(top_n))]]
        return {"peak_years": peak_years, "genres_ranked": ranked, "top_n": int(top_n)}


# ======================================================================================
# Testes automatizados internos
# ======================================================================================

def _mkraw(name: str, release: str, price: str, genres: str,
           dev: str = "", pub: str = "") -> dict[str, str]:
    return {
        "Name": name,
        "Release date": release,
        "Price": price,
        "Genres": genres,
        "Developers": dev,
        "Publishers": pub,
    }


def run_internal_tests(verbose: bool = False) -> bool:
    """Roda doctest e um unittest pequeno usando StringIO.

    Valida:
      - 2 jogos grátis e 3 pagos ⇒ 40.0% e 60.0%
      - Ano de pico e empate
    """
    # Doctest
    dres = doctest.testmod(verbose=False)
    if verbose:
        print(f"Doctest: {dres.failed} falhas de {dres.attempted} testes")

    class TestAggregations(unittest.TestCase):
        def setUp(self) -> None:
            # Monta um CSV em memória
            fieldnames = ["Name", "Release date", "Price", "Genres", "Developers", "Publishers"]
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            rows = [
                _mkraw("A", "2008", "0.00", "Action"),
                _mkraw("B", "Oct 21, 2008", "0", "RPG, Action"),
                _mkraw("C", "2009", "19.99", "Indie"),
                _mkraw("D", "2009-05-01", "abc", "Action"),  # preço inválido -> pago
                _mkraw("E", "2010", "10", ""),               # gêneros vazios agora são aceitos
            ]
            for r in rows:
                writer.writerow(r)
            self.csv_text = buf.getvalue()

            # Carrega com o mesmo parser
            buf2 = io.StringIO(self.csv_text)
            reader = csv.DictReader(buf2)
            games = []
            field_map = GameDataset(Path("dummy"))._build_field_map(list(reader.fieldnames or []))
            for row in reader:
                norm = GameDataset(Path("dummy"))._normalize_row(row, field_map)
                games.append(GameParser.parse(norm))
            self.analytics = GameAnalytics(games)

        def test_pct_free_paid(self):
            res = self.analytics.pct_free_vs_paid()
            self.assertEqual(res["counts"]["free"], 2)
            self.assertEqual(res["counts"]["paid"], 3)
            self.assertEqual(res["counts"]["total"], 5)
            self.assertAlmostEqual(res["free_pct"], 40.0, places=1)
            self.assertAlmostEqual(res["paid_pct"], 60.0, places=1)

        def test_years_with_tie(self):
            res = self.analytics.years_with_most_new_games()
            self.assertEqual(res["max_count"], 2)
            self.assertEqual(res["years"], [2008, 2009])

    suite = unittest.TestLoader().loadTestsFromTestCase(TestAggregations)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=(2 if verbose else 1)).run(suite)
    return bool(result.wasSuccessful() and dres.failed == 0)


# ======================================================================================
# Testes com amostra persistida
# ======================================================================================


class SampleTester:
    """Executa consultas na amostra e compara com expected_results.json.

    Se expected_results.json não existir, gera um template e encerra com instruções.
    """

    DEFAULT_TOP_N = 5

    def __init__(self, dataset: GameDataset, sample_dir: Path) -> None:
        self.dataset = dataset
        self.sample_dir = Path(sample_dir)
        self.sample_csv = self.sample_dir / "sample.csv"
        self.expected_json = self.sample_dir / "expected_results.json"

    @staticmethod
    def _float_close(a: float, b: float, tol: float = 0.05) -> bool:
        return abs(float(a) - float(b)) <= tol

    def _gen_template(self) -> None:
        template = {
            "pct_free_vs_paid": {
                "free_pct": None,
                "paid_pct": None,
                "counts": {"free": None, "paid": None, "total": None},
            },
            "years_with_most_new_games": {
                "max_count": None,
                "years": [],
            },
            "top_genres_in_peak_year": {
                "peak_years": [],
                "genres_ranked": [],
                "top_n": self.DEFAULT_TOP_N,
            },
        }
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        with self.expected_json.open("w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)

    def run(self, quiet: bool = False) -> bool:
        if not self.sample_csv.exists():
            raise DataValidationError(
                f"Amostra não encontrada em {self.sample_csv}. "
                f"Use --generate-sample primeiro."
            )
        games = self.dataset.load_from_file(self.sample_csv)
        analytics = GameAnalytics(games)

        # Se expected não existe, cria template e encerra
        if not self.expected_json.exists():
            self._gen_template()
            if not quiet:
                print(
                    f"Template de expectativas criado em '{self.expected_json}'.\n"
                    "Preencha os valores verdadeiros calculados manualmente e rode novamente "
                    "--run-sample-tests."
                )
            return False

        with self.expected_json.open("r", encoding="utf-8") as f:
            expected = json.load(f)

        # Calcula atual
        top_n = int(
            (expected.get("top_genres_in_peak_year") or {}).get("top_n", self.DEFAULT_TOP_N)
        )
        actual = {
            "pct_free_vs_paid": analytics.pct_free_vs_paid(),
            "years_with_most_new_games": analytics.years_with_most_new_games(),
            "top_genres_in_peak_year": analytics.top_genres_in_peak_year(top_n=top_n),
        }

        # Comparações com tolerância
        diffs: list[str] = []

        def cmp_dicts(path: str, exp: Any, act: Any) -> None:
            if isinstance(exp, dict) and isinstance(act, dict):
                keys = set(exp.keys()) | set(act.keys())
                for k in sorted(keys):
                    cmp_dicts(f"{path}.{k}", exp.get(k), act.get(k))
            elif isinstance(exp, list) and isinstance(act, list):
                n = min(len(exp), len(act))
                for i in range(n):
                    cmp_dicts(f"{path}[{i}]", exp[i], act[i])
                if len(exp) != len(act):
                    diffs.append(f"{path}: tamanhos diferem (exp={len(exp)} act={len(act)})")
            elif isinstance(exp, (int, float)) and isinstance(act, (int, float)):
                if not self._float_close(float(exp), float(act)):
                    diffs.append(f"{path}: exp={exp} act={act}")
            else:
                if exp != act:
                    diffs.append(f"{path}: exp={exp} act={act}")

        # Ajuste: se expected contém None, não comparar aquele campo (template incompleto)
        def prune_nones(obj: Any, act: Any) -> Any:
            if isinstance(obj, dict) and isinstance(act, dict):
                return {k: prune_nones(obj[k], act.get(k)) for k in obj if obj[k] is not None}
            return obj

        exp_pruned = {
            "pct_free_vs_paid": prune_nones(expected["pct_free_vs_paid"], actual["pct_free_vs_paid"]),
            "years_with_most_new_games": prune_nones(
                expected["years_with_most_new_games"], actual["years_with_most_new_games"]
            ),
            "top_genres_in_peak_year": prune_nones(
                expected["top_genres_in_peak_year"], actual["top_genres_in_peak_year"]
            ),
        }

        cmp_dicts("pct_free_vs_paid", exp_pruned["pct_free_vs_paid"], actual["pct_free_vs_paid"])
        cmp_dicts(
            "years_with_most_new_games",
            exp_pruned["years_with_most_new_games"],
            actual["years_with_most_new_games"],
        )
        cmp_dicts(
            "top_genres_in_peak_year",
            exp_pruned["top_genres_in_peak_year"],
            actual["top_genres_in_peak_year"],
        )

        if diffs:
            if not quiet:
                print("Diferenças encontradas entre expected_results.json e resultados atuais:")
                for d in diffs:
                    print(" -", d)
            return False
        else:
            if not quiet:
                print("Sample tests: OK (resultados batem com expected_results.json).")
            return True


# ======================================================================================
# Relatório Markdown
# ======================================================================================


class ReportBuilder:
    """Gera um relatório Markdown com respostas e breve discussão."""

    def build(self, out_path: Path, analytics: GameAnalytics, top_n: int = 5) -> Path:
        p1 = analytics.pct_free_vs_paid()
        p2 = analytics.years_with_most_new_games()
        p3 = analytics.top_genres_in_peak_year(top_n=top_n)

        def tbl_row(k: str, v: Any) -> str:
            return f"| {k} | {v} |"

        md = io.StringIO()
        md.write("# Games Analyzer — Relatório\n\n")
        md.write("## Metodologia\n")
        md.write(textwrap.dedent("""
        - **Parsing de dados**: datas em múltiplos formatos; fallback por regex de ano (`\\b(19|20)\\d{2}\\b`).
        - **Preço**: limpeza de símbolos/textos; vírgulas → ponto; palavras como *Free/Gratuito* → 0.0.
        - **Gratuito**: inferido por `Price == 0.0` (quando `price` válido). Preços inválidos contam como **não grátis**.
        - **Gêneros**: separados por vírgulas, com espaços aparados; vazios descartados.
        - **Linhas inválidas**: processamento *fail-soft* (apenas casos essenciais, como `Name` vazio).
        """).strip() + "\n\n")

        md.write("## Pergunta 1 — % grátis vs pagos\n\n")
        md.write("| Métrica | Valor |\n|---|---|\n")
        md.write(tbl_row("Jogos grátis", p1["counts"]["free"]) + "\n")
        md.write(tbl_row("Jogos pagos", p1["counts"]["paid"]) + "\n")
        md.write(tbl_row("Total", p1["counts"]["total"]) + "\n")
        md.write(tbl_row("% grátis", f"{p1['free_pct']:.1f}%") + "\n")
        md.write(tbl_row("% pagos", f"{p1['paid_pct']:.1f}%") + "\n\n")
        md.write(textwrap.dedent("""
        **Discussão**: a razão grátis/pago sugere estratégia de monetização (F2P, DLCs, cosméticos),
        além de efeitos de catálogo legado e promoções.
        """).strip() + "\n\n")

        md.write("## Pergunta 2 — Anos com mais lançamentos\n\n")
        if p2["years"]:
            years_str = ", ".join(str(y) for y in p2["years"])
            md.write(f"- **Pico**: {years_str} (máximo = {p2['max_count']})\n\n")
        else:
            md.write("- **Sem anos válidos** no dataset.\n\n")
        md.write(textwrap.dedent("""
        **Reflexão**: picos podem refletir fatores externos (geração de consoles, motores acessíveis,
        plataformas de distribuição digital, choques macroeconômicos).
        """).strip() + "\n\n")

        md.write("## Pergunta 3 — Gêneros nos anos de pico\n\n")
        if p3["peak_years"] and p3["genres_ranked"]:
            md.write(f"- Anos de pico: {', '.join(map(str, p3['peak_years']))}\n\n")
            md.write("| Rank | Gênero | Contagem |\n|---:|---|---:|\n")
            for i, item in enumerate(p3["genres_ranked"], start=1):
                md.write(f"| {i} | {item['genre']} | {item['count']} |\n")
            md.write("\n")
        else:
            md.write("- **Sem gêneros válidos** para anos de pico.\n\n")
        md.write(textwrap.dedent("""
        **Por quê isso importa?** Concentrar gêneros de maior tração nos anos de maior oferta ajuda
        decisões de portfólio, marketing e posicionamento competitivo.
        """).strip() + "\n")

        out_path = Path(out_path)
        out_path.write_text(md.getvalue(), encoding="utf-8")
        return out_path


# ======================================================================================
# CLI
# ======================================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    epilog = textwrap.dedent("""
    Exemplos:

      Gerar amostra e relatório:
        python games_analyzer.py --csv data/games.csv --generate-sample --report report.md

      Preparar template e rodar testes da amostra:
        python games_analyzer.py --csv data/games.csv --run-sample-tests

      Rodar as respostas no console:
        python games_analyzer.py --csv data/games.csv
    """).strip()
    p = argparse.ArgumentParser(
        description="Analisador de CSV de jogos (stdlib only).",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=Path, required=True, help="Caminho para o CSV principal.")
    p.add_argument("--generate-sample", action="store_true",
                   help="Gera sample/sample.csv uma única vez (exclui os 20 primeiros registros).")
    p.add_argument("--run-sample-tests", action="store_true",
                   help="Executa testes sobre sample/sample.csv comparando com sample/expected_results.json "
                        "(gera template se ausente).")
    p.add_argument("--run-internal-tests", action="store_true",
                   help="Roda doctest + unittest embutidos.")
    p.add_argument("--report", type=Path, default=None,
                   help="Gera report.md no caminho indicado.")
    p.add_argument("--top-n", type=int, default=5, help="Top-N de gêneros nos anos de pico (default: 5).")
    p.add_argument("--quiet", action="store_true", help="Silencia saídas informativas.")
    p.add_argument("--debug-invalid", action="store_true",
                   help="Exibe as linhas inválidas (número da linha e motivo) ao carregar o dataset completo.")
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    quiet = bool(args.quiet)
    vlog = (lambda *a, **k: None) if quiet else print

    try:
        dataset = GameDataset(args.csv)

        # Geração de amostra (se solicitado)
        if args.generate_sample:
            out = dataset.sample_once(Path("sample"))
            vlog(f"Amostra criada/retornada em: {out}")

        # Testes de amostra (se solicitado)
        if args.run_sample_tests:
            tester = SampleTester(dataset, Path("sample"))
            ok = tester.run(quiet=quiet)
            # não interrompe o restante automaticamente

        # Testes internos (se solicitado)
        if args.run_internal_tests:
            ok = run_internal_tests(verbose=not quiet)
            if not ok:
                vlog("Falhas nos testes internos.")
            else:
                vlog("Testes internos OK.")

        # Precisamos carregar o dataset completo?
        need_full_load = (not any([args.generate_sample, args.run_sample_tests, args.run_internal_tests])
                          or args.report)

        if need_full_load:
            games = dataset.load()
            if not quiet and dataset.invalid_rows_count:
                vlog(f"Aviso: {dataset.invalid_rows_count} linha(s) inválida(s) ignorada(s).")
                if args.debug_invalid:
                    # Mostra até 20 exemplos para não poluir
                    preview = dataset.invalid_rows_detail[:20]
                    if preview:
                        vlog("Exemplos de linhas inválidas (linha, motivo):")
                        for ln, reason in preview:
                            vlog(f" - L{ln}: {reason}")
                        if len(dataset.invalid_rows_detail) > 20:
                            vlog(f"... (+{len(dataset.invalid_rows_detail)-20} outras)")

            analytics = GameAnalytics(games)

            # Se nenhum modo especial e sem --report, imprime respostas no console
            if not any([args.generate_sample, args.run_sample_tests, args.run_internal_tests, args.report]):
                p1 = analytics.pct_free_vs_paid()
                p2 = analytics.years_with_most_new_games()
                p3 = analytics.top_genres_in_peak_year(top_n=args.top_n)

                vlog("Pergunta 1 — % grátis vs pagos")
                vlog(json.dumps(p1, ensure_ascii=False, indent=2))
                vlog("\nPergunta 2 — Anos com mais lançamentos")
                vlog(json.dumps(p2, ensure_ascii=False, indent=2))
                vlog("\nPergunta 3 — Gêneros nos anos de pico")
                vlog(json.dumps(p3, ensure_ascii=False, indent=2))

            # Relatório (se solicitado)
            if args.report:
                out = ReportBuilder().build(Path(args.report), analytics, top_n=args.top_n)
                vlog(f"Relatório gerado em: {out}")

        return 0
    except DataValidationError as e:
        if not quiet:
            print(f"[ERRO] {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        if not quiet:
            print(f"[ERRO] Arquivo não encontrado: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        if not quiet:
            print(f"[ERRO] Inesperado: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
