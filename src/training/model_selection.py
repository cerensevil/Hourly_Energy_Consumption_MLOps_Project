def select_best_model(results: dict):
    """
    Select best model using business-aware logic.

    Rules
    -----
    1. Baseline is safety anchor.
    2. Models worse than baseline CWE are rejected.
    3. Winner = lowest CWE among remaining models.
    4. If none beats baseline → baseline wins.
    """

    if "baseline" not in results:
        raise ValueError("Baseline model required for comparison.")

    baseline_cwe = results["baseline"]["cwe"]

    # ------------------------------------------------
    # Candidate models (better than baseline)
    # ------------------------------------------------

    candidates = {}

    for name, metrics in results.items():

        if name == "baseline":
            continue

        if metrics["cwe"] < baseline_cwe:
            candidates[name] = metrics

    # ------------------------------------------------
    # If no model beats baseline
    # ------------------------------------------------

    if not candidates:
        return "baseline", results["baseline"]

    # ------------------------------------------------
    # Winner = lowest CWE
    # ------------------------------------------------

    winner = min(candidates.items(), key=lambda x: x[1]["cwe"])

    return winner