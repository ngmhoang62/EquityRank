"""Lock the research model, rank without outcomes, and report results."""

import json
from hashlib import file_digest

import joblib
import numpy as np
import pandas as pd

from .backtest import cost_grid, execution_cost_grid, marked_book, position_returns, simulate_execution
from .baselines import RANDOM_SEEDS
from .diagnostics import paired_calendar_intervals
from .evaluation import evaluate_ranked_books, jpx_sharpe, rank_scores
from .final_pipeline import PRICE_INPUTS, atomic_write
from .liquidity import cost_sensitivity, simulate_ranked_portfolio
from .preprocessing import KEY, LABEL_END, TARGET, load_stock_prices
from .release_data import FIN_COLS, PRICE_COLS, TREND_COLS, build_panel
from .release_experiment import (
    CUTOFF, FINANCE, INTRADAY, PORTFOLIO, PROTOCOL, calendar_labels, choose_parameters, economic_scores,
    outage_signals, research_inputs, select_winner, selection_tables, signal_window, stable_random_scores,
    train_before, write_csv, write_json,
)
from .release_models import BASE, CANDIDATES, FEATURES, SPECS, fit_candidate, predict_candidate
from .splitting import development_windows

HISTORY_RULES = ["One-session reversal", "20-session momentum"]
RANK_FILES = ("predictions.csv", "ridge_predictions.csv", "control_scores.csv")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_identity(out, selection):
    checksums = {}
    for file in RANK_FILES:
        with (out / file).open("rb") as stream:
            checksums[file] = file_digest(stream, "sha256").hexdigest()
    return {key: selection[key] for key in ("protocol", "fingerprint", "name", "params")} | {"files": checksums}


def save_rank_identity(out, selection):
    write_json(out / "rank_identity.json", _rank_identity(out, selection))


def verify_rank_identity(out, selection):
    if read_json(out / "rank_identity.json") != _rank_identity(out, selection):
        raise ValueError("Saved ranks or controls differ from the locked run. Run rank again.")


def _importance(prices, disclosures, out, device, tree, ledgers):
    path = out / "importance.csv"
    panel, history, fit, labels, ends = research_inputs(prices, disclosures)
    records, coefficients = [], []
    for start, end in development_windows(str(CUTOFF.date())):
        day = str(start.date())
        signals = signal_window(panel, ends, start, end)
        training = train_before(fit, start)
        groups = tuple(signals.groupby("Date").indices.values())
        for name in (BASE, tree):
            entry = next(item for item in ledgers[day]["candidates"] if item["model"] == name)
            params = entry["params"]
            bundle = fit_candidate(name, training, params, history, device)
            baseline = predict_candidate(bundle, signals, history, device, params, PORTFOLIO)
            if name == BASE:
                linear = bundle["predictor"].steps[-1][1]
                coefficients.extend((day, key, val) for key, val in zip(bundle["columns"], linear.coef_))
            columns = bundle["columns"]
            families = {"OHLCV": PRICE_COLS, "Trend": TREND_COLS, "Financial": FIN_COLS}
            families = {g: [k for k in cols if k in columns] for g, cols in families.items()}
            families = {g: cols for g, cols in families.items() if cols}
            for group, subset in {**{col: [col] for col in columns}, **families}.items():
                scores = {"Original": baseline}
                for seed in (7, 19, 37):
                    rng = np.random.default_rng(seed)
                    perm = np.arange(len(signals))
                    for positions in groups:
                        perm[positions] = rng.permutation(positions)
                    changed = signals[[*KEY, *columns]].copy()
                    changed[subset] = signals[subset].to_numpy()[perm]
                    scores[str(seed)] = predict_candidate(bundle, changed, history, device, params, PORTFOLIO)
                _, metrics = evaluate_ranked_books(signals, pd.DataFrame(scores), labels)
                for metric in ("Spread", "Top", "Bottom"):
                    values = metrics.pivot(index="Date", columns="Model", values=metric).dropna()
                    for seed in (7, 19, 37):
                        change = (jpx_sharpe(values["Original"]) - jpx_sharpe(values[str(seed)])) if metric == "Spread" else (
                            10_000 * (values["Original"] - values[str(seed)]).mean()
                        )
                        records.append((day, name, group, "Permutation", seed, metric, change, len(values)))
            for group, subset in families.items():
                keep = [col for col in columns if col not in subset]
                if not keep:
                    scores = pd.Series(training[TARGET].mean(), index=signals.index)
                else:
                    from sklearn.base import clone
                    fitted = clone(bundle["predictor"]).fit(training[keep], training[TARGET])
                    scores = pd.Series(fitted.predict(signals[keep]), index=signals.index)
                _, metrics = evaluate_ranked_books(
                    signals, pd.DataFrame({"Original": baseline, "Reduced": scores}), labels,
                )
                values = metrics.pivot(index="Date", columns="Model", values="Spread").dropna()
                records.append((day, name, group, "Refit without group", 7, "Spread",
                                jpx_sharpe(values["Original"]) - jpx_sharpe(values["Reduced"]), len(values)))
        print(f"Importance {day}: completed", flush=True)
    write_csv(path, pd.DataFrame(records, columns=["Fold", "Model", "Feature", "Method", "Seed", "Metric", "Decrease", "Common sessions"]))
    write_csv(out / "ridge_coefficients.csv", pd.DataFrame(coefficients, columns=["Fold", "Feature", "Coefficient"]))


def _combine_folds(folders, file, *, date=False):
    return pd.concat([
        pd.read_csv(folder / file, parse_dates=["Date"] if date else None).assign(Fold=folder.name)
        for folder in folders
    ], ignore_index=True)


def _outage_table(daily, prices):
    ends, _ = calendar_labels(prices.loc[prices["Date"].lt(CUTOFF)])
    affected = outage_signals(ends)
    grid = daily.pivot(index="Date", columns="Model", values="Spread")
    base = grid[BASE]
    alternative = "Ridge without outage labels"
    paired = grid[[BASE, alternative]].dropna()
    cases = {
        "Ridge original": base.dropna(),
        "Ridge remove affected evaluation dates": base.loc[~base.index.isin(affected)].dropna(),
        "Ridge training pair": paired[BASE],
        "Ridge exclude affected training labels": paired[alternative],
    }
    return pd.DataFrame([(name, len(spread), jpx_sharpe(spread), 10_000 * spread.mean())
                         for name, spread in cases.items()],
                        columns=["Scenario", "Sessions", "JPX Sharpe", "Spread bps"])


def select(prices, disclosures, out, identity, device):
    folders = [out / "folds" / str(start.date()) for start, _ in development_windows(str(CUTOFF.date()))]
    ledgers = {folder.name: read_json(folder / "done.json") for folder in folders}
    if any(record["fingerprint"] != identity for record in ledgers.values()):
        raise ValueError("A development checkpoint does not match the current research inputs.")
    daily = _combine_folds(folders, "daily.csv", date=True)
    marks = _combine_folds(folders, "marks.csv", date=True)
    fold_map = _combine_folds(folders, "calendar.csv", date=True).set_index("Date")["Fold"]
    score_table, folds = selection_tables(daily, CANDIDATES, fold_map)
    winner = select_winner(score_table, CANDIDATES)
    priced = marks.assign(Net=lambda book: book["Gross"] * 10_000 - 5 * book["Turnover"])
    grid = priced.pivot(index="Date", columns="Model", values="Net")
    matched = {}
    for name in grid:
        common = grid[BASE].notna() & grid[name].notna()
        matched[name] = (grid[name].loc[common].mean(), grid[BASE].loc[common].mean(), int(common.sum()))
    match_columns = ["Net at 5 bps", "Ridge net on matched sessions", "Common cost sessions"]
    score_table = score_table.join(pd.DataFrame.from_dict(matched, orient="index", columns=match_columns), on="Model")
    score_table["Candidate"] = score_table["Model"].isin(CANDIDATES)
    score_table["Selected"] = score_table["Model"].eq(winner)
    score_table["Inputs"] = score_table["Model"].map({n: len(FEATURES[v[1]]) for n, v in SPECS.items()})
    score_table = score_table.sort_values(["JPX Sharpe", "Spread bps"], ascending=False)
    for name, table in {
        "daily": daily, "marks": marks, "summary": score_table, "folds": folds.reset_index(),
        "outage": _outage_table(daily, prices),
    }.items():
        write_csv(out / f"development_{name}.csv", table)
    for name in ("matched", "liquidity", "sensitivity", "execution", "execution_orders", "learning"):
        write_csv(out / f"development_{name}.csv", _combine_folds(folders, f"{name}.csv"))
    aligned = _combine_folds(folders, "matched.csv", date=True)
    aligned_table, _ = selection_tables(aligned, [*CANDIDATES, *HISTORY_RULES], fold_map)
    write_csv(out / "matched_summary.csv", aligned_table)
    grid = daily.pivot(index="Date", columns="Model", values="Spread").reindex(fold_map.index)
    write_csv(out / "sharpe_intervals.csv", paired_calendar_intervals(grid[list(CANDIDATES)], fold_map, BASE))
    _importance(prices, disclosures, out, device, FINANCE, ledgers)
    panel, history, fit, labels, ends = research_inputs(prices, disclosures)
    params, trials = choose_parameters(winner, fit, panel, history, labels, ends, device)
    write_json(out / "selected_model.json", {
        "protocol": PROTOCOL, "fingerprint": identity, "name": winner, "params": params, "baseline": BASE,
        "cutoff": str(CUTOFF.date()), "refit_search": trials, "candidates": list(CANDIDATES),
        "validation_score_sessions": int(score_table.loc[score_table["Selected"], "Common score sessions"].item()),
        "selection_rule": "validation Sharpe, spread, simplicity on shared sessions",
        "finance_coverage": next(iter(ledgers.values()))["finance_coverage"],
    })
    print(f"Selected: {winner}", flush=True)


def train_release(prices, disclosures, out, selection, device):
    _, history, fit, _, _ = research_inputs(prices, disclosures)
    names = {"ridge": (BASE, {"alpha": 10000}), "selected": (selection["name"], selection["params"])}
    fitted = {}
    for alias, (name, params) in names.items():
        if name not in fitted:
            fitted[name] = fit_candidate(name, fit, params, history, device)
        model = fitted[name]
        bundle = {"protocol": PROTOCOL, "name": name, "params": params, "model": model,
                  "fingerprint": selection["fingerprint"]}
        atomic_write(out / f"{alias}.joblib", lambda p: joblib.dump(bundle, p))
        write_json(out / f"{alias}_metadata.json", {
            "protocol": PROTOCOL, "name": name, "params": params, "columns": model["columns"],
            "last_label": str(fit[LABEL_END].max().date()), "rows": len(fit),
            "fingerprint": selection["fingerprint"], "seeds": list(model.get("states", {})),
        })


def rank_from_inputs(bundle, panel, history, device):
    if panel.empty:
        return rank_scores(panel, pd.Series(dtype=float, index=panel.index))
    score = predict_candidate(bundle["model"], panel, history, device, bundle["params"], PORTFOLIO)
    return rank_scores(panel, score)


def _read_scores(out, selection):
    verify_rank_identity(out, selection)
    base = pd.read_csv(out / "ridge_predictions.csv", parse_dates=["Date"], float_precision="round_trip")
    selected = pd.read_csv(out / "predictions.csv", parse_dates=["Date"], float_precision="round_trip")
    pairs = base.merge(selected, on=KEY, suffixes=("_ridge", "_selected"), validate="one_to_one", sort=False)
    if len(base) != len(selected) or len(pairs) != len(base):
        raise ValueError("Release models must rank the same eligible key set.")
    if selection["name"] == BASE:
        pd.testing.assert_frame_equal(base, selected)
        return {BASE: base}, pairs
    return {BASE: base, selection["name"]: selected}, pairs


def _historical_summary(daily, sessions):
    common_dates = daily.pivot(index="Date", columns="Model", values="Spread").dropna().index
    rows = []
    for model, frame in daily.groupby("Model"):
        common = frame.loc[frame["Date"].isin(common_dates)]
        rows.append((model, len(sessions), frame["Spread"].count(), len(common_dates),
                     jpx_sharpe(common["Spread"]), 10_000 * common["Spread"].mean(),
                     common["IC"].mean(), int(frame["Missing selected labels"].sum())))
    return pd.DataFrame(rows, columns=[
        "Model", "Sessions", "Scored sessions", "Common sessions", "JPX Sharpe", "Spread bps", "IC", "Missing labels",
    ])


def evaluate_release(prices, out, selection):
    books, pairs = _read_scores(out, selection)
    ends, labels = calendar_labels(prices)
    dates = ends.index[(ends.index >= CUTOFF) & ends.notna()]
    pairs = pairs.loc[pairs["Date"].isin(dates)].reset_index(drop=True)
    rules = pd.read_csv(out / "control_scores.csv", parse_dates=["Date"], float_precision="round_trip")
    pairs = pairs.merge(rules, on=KEY, how="left", validate="one_to_one")
    scores = pd.DataFrame({BASE: pairs["Score_ridge"], INTRADAY: pairs[INTRADAY],
                           **{f"Random {seed}": stable_random_scores(pairs, seed) for seed in RANDOM_SEEDS}})
    if selection["name"] != BASE:
        scores[selection["name"]] = pairs["Score_selected"]
    ranks, daily = evaluate_ranked_books(pairs[KEY], scores, labels)
    for model, saved in books.items():
        generated = ranks.loc[ranks["Model"].eq(model), [*KEY, "Rank"]].set_index(KEY)["Rank"]
        actual = saved.set_index(KEY)["Rank"].reindex(generated.index)
        pd.testing.assert_series_equal(generated, actual)
    write_csv(out / "historical_daily.csv", daily)
    write_csv(out / "historical_summary.csv", _historical_summary(daily, dates))
    matched = pairs[HISTORY_RULES].notna().all(axis=1)
    if matched.any():
        alternatives = scores.loc[matched].join(pairs.loc[matched, HISTORY_RULES])
        _, subset = evaluate_ranked_books(pairs.loc[matched, KEY], alternatives, labels)
        write_csv(out / "historical_matched_summary.csv", _historical_summary(subset, dates))


def backtest_release(prices, out, selection):
    ranks_by_model, _ = _read_scores(out, selection)
    ends, _ = calendar_labels(prices)
    dates = ends.index[(ends.index >= CUTOFF) & ends.notna()]
    mark = position_returns(prices)
    books = pd.concat([
        marked_book(ranks, prices, dates, returns=mark).assign(Model=name) for name, ranks in ranks_by_model.items()
    ], ignore_index=True)
    shared = books.pivot(index="Date", columns="Model", values="Gross").dropna().index
    write_csv(out / "backtest_daily.csv", books)
    results = []
    for name, book in books.groupby("Model"):
        native = cost_grid(book).assign(Model=name, Scope="Native")
        paired = cost_grid(book.assign(Gross=book["Gross"].where(book["Date"].isin(shared)))).assign(
            Model=name, Scope="Common", **{"Drawdown bps": np.nan})
        results.extend((native, paired))
    write_csv(out / "backtest_costs.csv", pd.concat(results, ignore_index=True))
    labels = prices[[*KEY, TARGET]]
    proxies = {
        name: simulate_ranked_portfolio(ranks.merge(labels, on=KEY, how="left", validate="one_to_one"),
                                        dates, PORTFOLIO)
        for name, ranks in ranks_by_model.items()
    }
    shared = pd.DataFrame({name: book["Gross bps"] for name, book in proxies.items()}).dropna().index
    proxy_results = [cost_sensitivity(book.loc[shared], (0, 2, 5, 10)).reset_index().assign(
        Model=name, **{"Common sessions": len(shared)}) for name, book in proxies.items()]
    write_csv(out / "target_weight_proxy.csv", pd.concat(proxy_results, ignore_index=True))
    outputs = {name: [] for name in ("execution_daily", "execution_orders", "execution_costs", "execution_coverage")}
    for name, ranks in ranks_by_model.items():
        trace, trades = simulate_execution(ranks, prices, dates)
        outputs["execution_daily"].append(trace.assign(Model=name))
        outputs["execution_orders"].append(trades.assign(Model=name))
        outputs["execution_costs"].append(execution_cost_grid(trace).assign(Model=name))
        outputs["execution_coverage"].append(pd.DataFrame([{
            "Model": name, "Signals": len(dates), "Orders": len(trades), "Filled": int(trace["Filled"].sum()),
            "Unfilled": int(trace["Unfilled"].sum()), "Blocked": int(trace["Blocked"].sum()),
            "Deferred": int(trace["Deferred today"].sum()), "Unresolved": int(trades["Status"].eq("open").sum()),
            "Execution sessions": len(trace), "Max deferred notional": trace["Deferred notional"].max(),
        }]))
    for name, tables in outputs.items():
        write_csv(out / f"{name}.csv", pd.concat(tables, ignore_index=True))


def run_step(command, raw, out, identity, device):
    if command == "rank":
        prices = pd.concat([load_stock_prices(raw / group / "stock_prices.csv", columns=PRICE_INPUTS)
                            for group in ("train_files", "supplemental_files")], ignore_index=True)
    else:
        prices = pd.read_pickle(out / "prices.pkl")
    if command == "select":
        select(prices, pd.read_pickle(out / "financial.pkl"), out, identity, device)
        return
    selection = read_json(out / "selected_model.json")
    if selection["protocol"] != PROTOCOL or selection["fingerprint"] != identity:
        raise ValueError("The selected configuration does not match current inputs.")
    if command in {"train", "rank"}:
        disclosures = pd.read_pickle(out / "financial.pkl")
    if command == "train":
        train_release(prices, disclosures, out, selection, device)
    elif command == "rank":
        panel, history = build_panel(prices[PRICE_INPUTS], disclosures)
        panel = panel.loc[panel["Date"].ge(CUTOFF)]
        for alias, file, expected in (("ridge", "ridge_predictions.csv", BASE),
                                       ("selected", "predictions.csv", selection["name"])):
            bundle = joblib.load(out / f"{alias}.joblib")
            params = selection["params"] if alias == "selected" else {"alpha": 10000}
            if (bundle.get("protocol"), bundle["fingerprint"], bundle["name"], bundle["params"]) != (
                PROTOCOL, identity, expected, params
            ):
                raise ValueError(f"{alias} model belongs to a different configuration.")
            write_csv(out / file, rank_from_inputs(bundle, panel, history, device))
        write_csv(out / "control_scores.csv", panel[KEY].join(economic_scores(panel)))
        save_rank_identity(out, selection)
    elif command == "evaluate":
        evaluate_release(prices, out, selection)
    elif command == "backtest":
        backtest_release(prices, out, selection)
    print(f"{command}: completed", flush=True)
