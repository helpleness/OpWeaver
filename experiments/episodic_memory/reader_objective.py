"""Reader targets; uncertainty belongs to the availability/controller objective.

This does not teach a generative model to explain uncertainty. The legacy option
exists solely to reproduce historical checkpoints that used a fixed refusal.
"""

from .generate_data import FIELDS, TASKS
from .semantic_data import ANSWERS


def reader_queries(data, legacy_refusal_target=False):
    return [
        q
        for q in data["queries"]
        if q["field"] < 6
        and (
            legacy_refusal_target
            or all(
                data["events"][q["event"] + view]["values"][TASKS[q["field"]]] >= 0
                for view in range(2)
            )
        )
    ]


def reader_batch(data, queries, batch, rng, legacy_refusal_target=False):
    if not queries:
        raise ValueError("No eligible reader examples")
    qs = [rng.choice(queries) for _ in range(batch)]
    ids = [q["event"] + rng.randrange(2) for q in qs]
    present = [rng.random() > 0.15 for _ in qs] if legacy_refusal_target else [True] * batch
    answers = []
    for q, i, keep in zip(qs, ids, present):
        field = q["field"]
        value = data["events"][i]["values"][TASKS[field]]
        if keep and value >= 0:
            answers.append(rng.choice(ANSWERS[field]).format(FIELDS[TASKS[field]][value]))
        elif legacy_refusal_target:
            answers.append("我没有这段经历的记忆。")
        else:
            raise ValueError("Default reader objective requires an available fact")
    return qs, ids, present, answers
