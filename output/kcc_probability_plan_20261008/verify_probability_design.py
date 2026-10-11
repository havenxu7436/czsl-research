"""Check the proposed probability identities, not model performance or runtime."""
import json
import math
from pathlib import Path
import random


def logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(v - maximum) for v in values))


def distribution(scores, temperature):
    row_normalizers = [logsumexp([v / temperature for v in row]) for row in scores]
    total_normalizer = logsumexp(row_normalizers)
    objects = [math.exp(z - total_normalizer) for z in row_normalizers]
    attributes = [[math.exp(v / temperature - z) for v in row]
                  for row, z in zip(scores, row_normalizers)]
    return objects, attributes


def readout(base, delta, temperature):
    if all(value == 0 for row in delta for value in row):
        return [row[:] for row in base]
    rows = []
    for old, shift in zip(base, delta):
        changed = [s + d for s, d in zip(old, shift)]
        correction = temperature * (
            logsumexp([v / temperature for v in changed])
            - logsumexp([v / temperature for v in old]))
        rows.append([v - correction for v in changed])
    return rows


def flatten(matrix):
    return [v for row in matrix for v in row]


def main():
    generator = random.Random(20261008)
    maximum_mass_error = 0.0
    maximum_joint_error = 0.0
    maximum_constant_shift_error = 0.0
    cases = 0
    for n_objects, n_attributes in [(12, 16), (3, 5), (1, 4)]:
        for temperature in [0.05, 0.2, 1.0]:
            for spread in [0.1, 1.0, 50.0]:
                base = [[generator.uniform(-spread, spread) for _ in range(n_attributes)]
                        for _ in range(n_objects)]
                zeros = [[0.0] * n_attributes for _ in range(n_objects)]
                assert readout(base, zeros, temperature) == base
                delta = [[generator.uniform(-0.1, 0.1) for _ in range(n_attributes)]
                         for _ in range(n_objects)]
                obj_base, attr_base = distribution(base, temperature)
                changed = readout(base, delta, temperature)
                obj_new, attr_new = distribution(changed, temperature)
                perturbed = [[s + d for s, d in zip(old, shift)]
                             for old, shift in zip(base, delta)]
                _, attr_target = distribution(perturbed, temperature)
                for o in range(n_objects):
                    maximum_mass_error = max(maximum_mass_error, abs(obj_new[o] - obj_base[o]))
                    for a in range(n_attributes):
                        expected = obj_base[o] * attr_target[o][a]
                        actual = obj_new[o] * attr_new[o][a]
                        maximum_joint_error = max(maximum_joint_error, abs(expected - actual))
                assert abs(sum(obj_new) - 1.0) < 1e-11
                assert all(abs(sum(row) - 1.0) < 1e-11 for row in attr_new)
                # An object-only constant cannot modify its attribute distribution.
                constant_delta = [[generator.uniform(-0.1, 0.1)] * n_attributes
                                  for _ in range(n_objects)]
                constant_readout = readout(base, constant_delta, temperature)
                maximum_constant_shift_error = max(maximum_constant_shift_error,
                    max(abs(a - b) for a, b in zip(flatten(base), flatten(constant_readout))))
                cases += 1
    assert maximum_mass_error < 1e-11
    assert maximum_joint_error < 1e-11
    assert maximum_constant_shift_error < 1e-11
    result = dict(
        passed=True,
        scope="Synthetic float64 probability identities only; no model training or VAL/TEST inference",
        cases=cases,
        zero_branch_exact_score_replay=True,
        normalized_joint_and_conditionals=True,
        maximum_object_marginal_error=maximum_mass_error,
        maximum_joint_identity_error=maximum_joint_error,
        maximum_object_constant_shift_error=maximum_constant_shift_error,
        gradient_checked=False,
        native_evaluator_checked=False,
        new_model_performance_known=False,
    )
    output = Path(__file__).with_name("probability_identity_checks.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
