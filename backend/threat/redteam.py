
import math


def redteam(graph, max_depth=4):
    """
    Analyze potentially damaging paths through a directed
    infrastructure dependency graph.

    Supports:
    - Existing strength and lag-based ranking.
    - Optional compromise_cost and damage edge attributes.
    - Cycle prevention.
    - Maximum path depth.
    - Top 10 ranked attack paths.

    Cost and damage are only evaluated when supplied by
    the graph. They are never invented.
    """

    if not isinstance(graph, dict):
        return []

    if not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")

    edges = graph.get("edges", [])

    if not isinstance(edges, list):
        return []

    adjacency = {}

    for edge in edges:
        if not isinstance(edge, dict):
            continue

        src = edge.get("src")
        dst = edge.get("dst")

        if not src or not dst:
            continue

        try:
            strength = float(edge.get("strength", 0))
            lag = float(edge.get("lag_min", 20))
        except (TypeError, ValueError):
            continue

        if not math.isfinite(strength) or not math.isfinite(lag):
            continue

        if not 0 <= strength <= 1 or lag <= 0:
            continue

        # Optional R3 attributes.
        cost = edge.get("compromise_cost")
        damage = edge.get("damage")

        try:
            if cost is not None:
                cost = float(cost)
                if not math.isfinite(cost) or cost < 0:
                    cost = None

            if damage is not None:
                damage = float(damage)
                if not math.isfinite(damage) or damage < 0:
                    damage = None

        except (TypeError, ValueError):
            cost = None
            damage = None

        adjacency.setdefault(src, []).append({
            "node": dst,
            "strength": strength,
            "lag": lag,
            "cost": cost,
            "damage": damage
        })

    results = []

    def explore(
        node,
        path,
        score,
        total_lag,
        total_cost,
        total_damage,
        all_costs_known,
        all_damage_known
    ):

        if len(path) > 1:

            result = {
                "entry_node": path[0],
                "path": path,
                "score": round(score, 4),
                "total_lag_min": round(total_lag, 2),
                "damage_node": path[-1]
            }

            # Only calculate cost/damage ranking when every
            # edge in this path provides both attributes.
            if all_costs_known and all_damage_known:
                average_strength = score / max(
                    sum(
                        edge["strength"] / edge["lag"]
                        for edge in path_edges
                    ),
                    1e-12
                ) if False else None

                result["total_compromise_cost"] = round(
                    total_cost, 4
                )
                result["estimated_damage"] = round(
                    total_damage, 4
                )

                # Prototype heuristic: higher damage per unit
                # of compromise cost ranks higher.
                result["risk_score"] = round(
                    total_damage / max(total_cost, 1e-9),
                    4
                )

            results.append(result)

        if len(path) - 1 >= max_depth:
            return

        for edge in adjacency.get(node, []):

            next_node = edge["node"]

            if next_node in path:
                continue

            edge_score = edge["strength"] / edge["lag"]

            explore(
                next_node,
                path + [next_node],
                score + edge_score,
                total_lag + edge["lag"],
                total_cost + (edge["cost"] or 0),
                total_damage + (edge["damage"] or 0),
                all_costs_known and edge["cost"] is not None,
                all_damage_known and edge["damage"] is not None
            )

    for entry in adjacency:

        explore(
            entry,
            [entry],
            0,
            0,
            0,
            0,
            True,
            True
        )

    # Use the richer risk score only when all candidate
    # paths have comparable cost and damage information.
    if results and all(
        "risk_score" in result for result in results
    ):
        results.sort(
            key=lambda item: item["risk_score"],
            reverse=True
        )
    else:
        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

    return results[:10]
