"""R5: within-teacher/type centered ridge, with the absolute fit's scaling."""
import numpy as np
from . import compact

METHODS = ('absolute', 'relative')
VARIANTS = ('uniform', 'absolute', 'relative', 'permuted_absolute', 'permuted_relative')
POLICIES = ('uniform', 'absolute', 'relative', 'permuted_absolute', 'permuted_relative',
            'measured', 'permuted_measured')
TYPE_NAMES = ('OI', 'OE', 'AE')


def fit_relative(xs, ys, types, alpha, normalization):
    """Min sum_jt ||P_t(Z_jt beta-y_jt/a_t)||^2 + alpha ||beta||^2.

    Center each teacher/type independently, including the design rows. Keep the
    original unaveraged squared-error sum; do not change type or teacher weights.
    The three intercept columns vanish, and their ridge optimum is exactly zero.
    """
    if alpha <= 0:
        raise ValueError('Ridge alpha must be positive')
    types = np.asarray(types, dtype=np.int64)
    center, spread, scales = [np.asarray(normalization[k], dtype=np.float64)
                              for k in ('center', 'spread', 'scales')]
    zz, yy = [], []
    for x, y in zip(xs, ys):
        z = compact.design(np.asarray(x, dtype=np.float64), types, center, spread)
        q = np.asarray(y, dtype=np.float64) / scales[types]
        for t in range(3):
            keep = types == t
            zz.append(z[keep] - z[keep].mean(0))
            yy.append(q[keep] - q[keep].mean())
    z, y = np.concatenate(zz), np.concatenate(yy)
    beta = np.linalg.solve(z.T @ z + float(alpha)*np.eye(z.shape[1]), z.T @ y)
    return dict(beta=beta, center=center, spread=spread, scales=scales, alpha=float(alpha),
                dimension=int(z.shape[1]), fit_folds=len(xs), intercepts_penalized=True,
                objective='within_teacher_type_centered_ridge', effective_coefficients=24)


def paired_models(xs, ys, types, alpha):
    absolute = compact.fit_ridge(xs, ys, types, alpha)
    normal = {k: absolute[k] for k in ('center', 'spread', 'scales')}
    return dict(absolute=absolute, relative=fit_relative(xs, ys, types, alpha, normal))


def centered_metrics(pred, target, types):
    p, y, t = np.asarray(pred), np.asarray(target), np.asarray(types)
    pc = np.concatenate([p[t == k]-p[t == k].mean() for k in range(3)])
    yc = np.concatenate([y[t == k]-y[t == k].mean() for k in range(3)])
    result = compact.point_metrics(pc, yc)
    result['r2'] = float(1-np.mean((pc-yc)**2)/np.mean(yc**2)) if np.mean(yc**2) > 0 else None
    return result
