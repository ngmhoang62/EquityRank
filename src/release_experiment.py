"""Walk-forward development on the research calendar."""

import hashlib
import json
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import numpy as np
import pandas as pd

from .backtest import execution_cost_grid, marked_book, position_returns, simulate_execution
from .baselines import RANDOM_SEEDS
from .evaluation import evaluate_ranked_books, jpx_sharpe, rank_weights
from .final_pipeline import atomic_write
from .liquidity import liquidity_portfolios
from .preprocessing import KEY, LABEL_END, TARGET
from .release_data import RATIOS, build_panel
from .release_models import BASE, FINANCE, FINANCE_CONFIG, NEURAL, SPECS, TREES, fit_candidate, predict_candidate
from .splitting import development_windows

CUTOFF = pd.Timestamp("2021-12-09")
PORTFOLIO = 200
PROTOCOL = "equity-rank-research-v2"
OUTAGE = pd.Timestamp("2020-10-01")
INTRADAY = "Intraday reversal"
FILL_MODELS = (BASE, FINANCE)


def write_json(path, data):
    atomic_write(path, lambda file: file.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"))


def write_csv(path, table):
    atomic_write(path, lambda file: table.to_csv(file, index=False))


def source_hash():
    digest = hashlib.sha256()
    project = Path(__file__).resolve().parents[1]
    files = [*(project / "src").rglob("*.py"), project / "scripts/run_jpx_project.py", FINANCE_CONFIG]
    for file in sorted(files):
        digest.update(file.relative_to(project).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def calendar_labels(prices):
    dates = pd.DatetimeIndex(prices["Date"].unique()).sort_values()
    ends = pd.Series(dates, index=dates).shift(-2)
    return ends, prices[[*KEY, TARGET]].assign(**{LABEL_END: prices["Date"].map(ends)})


def stable_random_scores(signals, seed):
    keys = pd.util.hash_pandas_object(signals[KEY], index=False).to_numpy(dtype=np.uint64)
    hashed = keys ^ np.uint64(seed)
    hashed = (hashed ^ (hashed >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    hashed = (hashed ^ (hashed >> 27)) * np.uint64(0x94D049BB133111EB)
    hashed ^= hashed >> 31
    return pd.Series((hashed >> 11) * 2.0 ** -53, index=signals.index)


def economic_scores(signals):
    return pd.DataFrame({INTRADAY: -signals["intraday_return"],
                         "One-session reversal": -signals["return_1"], "20-session momentum": signals["return_20"]})


def outage_signals(ends):
    dates = pd.Series(ends.index, index=ends.index)
    return ends.index[ends.eq(OUTAGE) | dates.shift(-1).eq(OUTAGE)]


def research_inputs(prices, disclosures, *, legacy_finance=False):
    prices = prices.loc[prices["Date"].lt(CUTOFF)]
    ends, labels = calendar_labels(prices)
    panel, history = build_panel(prices, disclosures, legacy_finance=legacy_finance)
    usable = np.isfinite(labels[TARGET]) & labels[LABEL_END].notna()
    fit = panel.merge(labels.loc[usable], on=KEY, validate="one_to_one")
    return panel, history, fit, labels, ends


def train_before(fit, start):
    return fit.loc[fit[LABEL_END].lt(start)]


def signal_window(panel, ends, start, end):
    dates = ends.index[(ends.index >= start) & ends.lt(end)]
    return panel.loc[panel["Date"].isin(dates)]


def choose_parameters(name, fit, panel, history, labels, ends, device):
    options = SPECS[name][2]
    if len(options) == 1:
        return options[0], []
    dates = pd.DatetimeIndex(fit["Date"].unique()).sort_values()[-120:]
    if len(dates) != 120:
        raise ValueError("Nested search requires 120 train dates.")
    cutoff = fit[LABEL_END].max() + pd.Timedelta(days=1)
    scores_by_window = []
    rounds = [[] for _ in options]
    for start, end in ((dates[0], dates[60]), (dates[60], cutoff)):
        inner_fit = train_before(fit, start)
        signals = signal_window(panel, ends, start, end)
        predictions = {}
        shared_guard = fit_candidate(name, inner_fit, options[0], history, device) if name == "Ridge short guard" else None
        for index, params in enumerate(options):
            validation = fit.loc[fit["Date"].ge(start) & fit[LABEL_END].lt(end)] if params.get("early_stop") else None
            model = shared_guard or fit_candidate(name, inner_fit, params, history, device, validation=validation)
            if "best_iteration" in model:
                rounds[index].append(model["best_iteration"])
            predictions[str(index)] = predict_candidate(model, signals, history, device, params, PORTFOLIO)
        _, daily = evaluate_ranked_books(signals, pd.DataFrame(predictions), labels, portfolio_size=PORTFOLIO)
        scores_by_window.append(daily)
    common = pd.concat(scores_by_window).pivot(index="Date", columns="Model", values="Spread").dropna()
    results = [jpx_sharpe(common[str(index)]) for index in range(len(options))]
    finite = np.isfinite(results)
    winner = int(np.argmax(np.where(finite, results, -np.inf)))
    trials = [{"model": name, "params": p, "sharpe": float(v) if np.isfinite(v) else None,
               "sessions": len(common), "fit_iterations": iterations,
               "status": "scored" if np.isfinite(v) else "undefined Sharpe"}
              for p, v, iterations in zip(options, results, rounds)]
    chosen = dict(options[winner])
    if not finite.any():
        trials[0]["status"] = "default: no candidate has a finite Sharpe"
    if chosen.pop("early_stop", False):
        chosen["n_estimators"] = int(np.median(rounds[winner]))
    return chosen, trials


def _short_traces(ranks, labels):
    count = ranks.groupby("Date")["Rank"].transform("size")
    short = ranks.loc[count.ge(2 * PORTFOLIO) & ranks["Rank"].ge(count - PORTFOLIO)].copy()
    short["Depth"] = count.loc[short.index] - 1 - short["Rank"]
    short = short.merge(labels[[*KEY, TARGET]], on=KEY, validate="one_to_one", how="left")
    market = ranks[KEY].merge(labels[[*KEY, TARGET]], on=KEY, how="left").groupby("Date")[TARGET].mean()
    short["Contribution"] = -(short[TARGET] - short["Date"].map(market)) * rank_weights(PORTFOLIO)[short["Depth"]]
    return short


def _portfolio_checks(name, signals, scores, primary, daily, labels, prices, dates, detail):
    scored = primary.merge(signals[[*KEY, "Liquidity", "beta_60"]], on=KEY, validate="one_to_one")
    scored = scored.merge(labels[[*KEY, TARGET]], on=KEY, how="left", validate="one_to_one")
    books = liquidity_portfolios(scored, dates).reset_index()
    detail["liquidity"].append(books.assign(Model=name))
    execution, orders = simulate_execution(primary, prices, dates)
    detail["execution"].append(execution_cost_grid(execution).assign(
        Model=name, Orders=len(orders), Filled=int(execution["Filled"].sum()),
        Deferred=int(execution["Deferred today"].sum()), Unresolved=int(orders["Status"].eq("open").sum()),
    ))
    order_stats = orders.merge(signals[[*KEY, "Liquidity"]], on=KEY, validate="one_to_one")
    detail["execution_orders"].append(order_stats.assign(Model=name))
    count = scored.groupby("Date")["Rank"].transform("size")
    long = scored["Rank"].lt(PORTFOLIO)
    short = scored["Rank"].ge(count - PORTFOLIO)
    active = scored.loc[count.ge(2 * PORTFOLIO) & (long | short)]
    depth = np.where(long.loc[active.index], active["Rank"], count.loc[active.index] - 1 - active["Rank"])
    beta_weight = rank_weights(PORTFOLIO)[depth] * np.where(long.loc[active.index], 1., -1.)
    beta = (active["beta_60"] * beta_weight).groupby(active["Date"]).sum(min_count=2 * PORTFOLIO)
    daily["Net beta"] = daily["Date"].map(beta)
    for size, ratio in ((100, 2.), (300, 2.), (200, 1.)):
        _, case = evaluate_ranked_books(signals, scores[[name]], labels, portfolio_size=size, weight_ratio=ratio)
        detail["sensitivity"].append(case.assign(**{"Portfolio size": size, "Weight ratio": ratio}))


def fold_run(panel, history, fit, labels, ends, prices, marks, start, end, enabled, device):
    train = train_before(fit, start)
    signals = signal_window(panel, ends, start, end)
    dates = ends.index[(ends.index >= start) & ends.lt(end)]
    rules = economic_scores(signals)
    simple_scores = {INTRADAY: rules[INTRADAY],
                     **{f"Random {seed}": stable_random_scores(signals, seed) for seed in RANDOM_SEEDS}}
    daily_tables, costs, ledger, short_positions = [], [], [], {}
    detail = {name: [] for name in ("matched", "liquidity", "execution", "execution_orders", "sensitivity", "learning")}
    matched = rules[["One-session reversal", "20-session momentum"]].notna().all(axis=1)
    dispersion_cut = train.groupby("Date")["market_dispersion"].first().median()
    dispersion = signals.groupby("Date")["market_dispersion"].first()
    for name in (*enabled, *simple_scores):
        started = perf_counter()
        if name in simple_scores:
            params, trials = {}, []
            scores = pd.DataFrame({name: simple_scores[name]})
        else:
            params, trials = choose_parameters(name, train, panel, history, labels, ends, device)
            training = train
            if name == "Ridge without outage labels":
                training = train.loc[~train["Date"].isin(outage_signals(ends))]
            fitted = fit_candidate(name, training, params, history, device)
            if fitted.get("learning_curve"):
                detail["learning"].append(pd.DataFrame(fitted["learning_curve"]).assign(Model=name))
            values = predict_candidate(fitted, signals, history, device, params, PORTFOLIO, seed_scores=True)
            if isinstance(values, tuple):
                main, seeds = values
                scores = seeds.rename(columns=lambda seed: f"{name} seed {seed}").assign(**{name: main})
            else:
                scores = pd.DataFrame({name: values})
        ranks, daily = evaluate_ranked_books(signals, scores, labels, diagnostics=True)
        daily["Market dispersion"] = daily["Date"].map(dispersion)
        daily["Dispersion regime"] = np.where(daily["Market dispersion"].isna(), "unknown",
                                              np.where(daily["Market dispersion"].gt(dispersion_cut), "high", "low"))
        primary = ranks.loc[ranks["Model"].eq(name)]
        costs.append(marked_book(primary, prices, dates, returns=marks).assign(Model=name))
        daily_tables.append(daily)
        if matched.any():
            _, subset = evaluate_ranked_books(signals.loc[matched], scores.loc[matched, [name]], labels)
            detail["matched"].append(subset)
        if name in FILL_MODELS:
            _portfolio_checks(name, signals, scores, primary, daily, labels, prices, dates, detail)
        if name in {BASE, "Ridge short guard"}:
            short_positions[name] = _short_traces(primary, labels)
        ledger.append({"model": name, "params": params, "trials": trials, "seconds": perf_counter() - started})
        print(f"{start:%Y-%m-%d} {name}: {ledger[-1]['seconds']:.1f}s", flush=True)
    if matched.any():
        _, case = evaluate_ranked_books(signals.loc[matched], rules.loc[matched].drop(columns=INTRADAY), labels)
        detail["matched"].append(case)
    base, guard = (short_positions.get(name) for name in (BASE, "Ridge short guard"))
    if base is not None and guard is not None:
        keys = guard[KEY].assign(Kept=True)
        moved = base.merge(keys, on=KEY, how="left")
        lost = moved.loc[moved["Kept"].isna() & moved["Contribution"].gt(0)].groupby("Date")["Contribution"].sum()
        complete = base.groupby("Date")["Contribution"].apply(lambda values: values.notna().all())
        book = next(table for table in costs if table["Model"].iloc[0] == "Ridge short guard")
        book["Foregone short contribution"] = book["Date"].map(lost).fillna(0.).where(book["Date"].map(complete).eq(True))
    extra = {name: pd.concat(tables, ignore_index=True) for name, tables in detail.items() if tables}
    extra["calendar"] = pd.DataFrame({"Date": dates})
    return pd.concat(daily_tables), pd.concat(costs), ledger, extra


def develop(prices, disclosures, output, fingerprint, device, *, pilot=False):
    windows = development_windows(str(CUTOFF.date()))
    windows = windows[:1] if pilot else windows
    mode = output / ("pilot" if pilot else "folds")
    pending = []
    for start, end in windows:
        folder = mode / str(start.date())
        done = folder / "done.json"
        checkpoint = json.loads(done.read_text(encoding="utf-8")) if done.exists() else {}
        if checkpoint.get("fingerprint") == fingerprint and all(
            (folder / file).is_file() for file in checkpoint.get("files", ["missing"])
        ):
            print(f"{start.date()} checkpoint validated", flush=True)
            continue
        pilot_folder = output / "pilot" / folder.name
        pilot_done = pilot_folder / "done.json"
        if not pilot and pilot_done.is_file():
            trial = json.loads(pilot_done.read_text(encoding="utf-8"))
            if trial["fingerprint"] == fingerprint and all((pilot_folder / f).is_file() for f in trial["files"]):
                for file in trial["files"]:
                    atomic_write(folder / file, lambda temp: copyfile(pilot_folder / file, temp))
                write_json(done, trial)
                print(f"{start.date()} validated pilot reused", flush=True)
                continue
        pending.append((start, end, folder, done))
    if not pending:
        return
    panel, history, fit, labels, ends = research_inputs(prices, disclosures, legacy_finance=True)
    pretest_prices = prices.loc[prices["Date"].lt(CUTOFF)]
    marks = position_returns(pretest_prices)
    finance_coverage = panel[list(RATIOS)].notna().all(axis=1).mean()
    print(f"Disclosure coverage {finance_coverage:.4f}; torch device {device}", flush=True)
    for start, end, folder, done in pending:
        daily, marks_day, ledger, extra = fold_run(
            panel, history, fit, labels, ends, pretest_prices, marks, start, end, SPECS, device,
        )
        tables = {"daily": daily, "marks": marks_day, **extra}
        files = [f"{name}.csv" for name in tables]
        for name, table in tables.items():
            write_csv(folder / f"{name}.csv", table)
        write_json(done, {"fingerprint": fingerprint, "candidates": ledger,
                          "finance_coverage": finance_coverage, "files": files})


def selection_tables(daily, candidates, fold_map):
    spread = daily.pivot(index="Date", columns="Model", values="Spread").reindex(fold_map.index)
    comparable = spread.loc[spread[list(candidates)].notna().all(axis=1)]
    fold = comparable.groupby(fold_map).agg(lambda s: jpx_sharpe(s.dropna()))
    rows = []
    for name in spread:
        available = daily.loc[daily["Model"].eq(name)]
        matched = comparable[[BASE, name]].dropna() if name != BASE else comparable[[BASE]].dropna()
        series = matched[name]
        rows.append({
            "Model": name, "JPX Sharpe": jpx_sharpe(series), "Spread bps": 10_000 * series.mean(),
            "IC": available.loc[available["Date"].isin(matched.index), "IC"].mean(),
            "Wins vs Ridge": int((fold[name] > fold[BASE]).sum()),
            "Common score sessions": len(matched), "Native score sessions": int(available["Spread"].notna().sum()),
        })
    return pd.DataFrame(rows), fold


def select_winner(table, candidates):
    eligible = table.loc[table["Model"].isin(candidates) & np.isfinite(table["JPX Sharpe"])].copy()
    if eligible.empty:
        raise ValueError("No candidate has a finite validation Sharpe on the shared calendar.")
    complexity = {name: (3 if family in NEURAL else 2 if family in {"guard", "ranker"} else
                        1 if family in TREES else 0) for name, (family, _, _) in SPECS.items()}
    eligible["Complexity"] = eligible["Model"].map(complexity)
    ranked = eligible.sort_values(["JPX Sharpe", "Spread bps", "Complexity", "Model"],
                                  ascending=[False, False, True, True])
    return ranked["Model"].iloc[0]
