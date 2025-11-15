import os, textwrap
from pathlib import Path

# pastikan PROJECT_ROOT sudah ada dari cell awal
print("PROJECT_ROOT      =", PROJECT_ROOT)

# Pakai nama file baru biar 100% nggak kebawa file lama
TRANSFORM_MODULE = os.path.join(PROJECT_ROOT, "pipeline", "preprocessing_clean.py")
print("TRANSFORM_MODULE  =", TRANSFORM_MODULE)

Path(os.path.dirname(TRANSFORM_MODULE)).mkdir(parents=True, exist_ok=True)

MODULE_CODE = textwrap.dedent(r"""
import tensorflow as tf
import tensorflow_transform as tft

# Kolom Adult dataset
NUMERIC_FEATURE_KEYS = [
    "age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"
]
CATEGORICAL_FEATURE_KEYS = [
    "workclass","education","marital-status","occupation",
    "relationship","race","sex","native-country"
]
LABEL_KEY = "income"  # '>50K' atau '<=50K'

def _xf(name):
    return name + "_xf"

def _dense_float(x):
    # [batch,1] -> [batch], cast float32, NaN -> 0
    x = tf.squeeze(x, axis=1)
    x = tf.cast(x, tf.float32)
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    return x

def _clean_str(x):
    # [batch,1] -> [batch], trim+lower, kosong -> 'unknown'
    x = tf.squeeze(x, axis=1)
    x = tf.strings.lower(tf.strings.strip(x))
    x = tf.where(tf.equal(x, ""), tf.constant("unknown"), x)
    return x

def preprocessing_fn(inputs):
    outputs = {}

    # numeric: impute + z-score
    for name in NUMERIC_FEATURE_KEYS:
        x = _dense_float(inputs[name])
        x = tft.scale_to_z_score(x)
        outputs[_xf(name)] = x

    # categorical: string clean + vocab -> id
    for name in CATEGORICAL_FEATURE_KEYS:
        s = _clean_str(inputs[name])
        ids = tft.compute_and_apply_vocabulary(
            s, top_k=None, num_oov_buckets=1, vocab_filename=name
        )
        outputs[_xf(name)] = ids

    # label: '>50k' -> 1, lainnya -> 0
    y = tf.squeeze(inputs[LABEL_KEY], axis=1)
    y = tf.strings.lower(tf.strings.strip(y))
    y = tf.cast(tf.equal(y, ">50k"), tf.int64)
    outputs[_xf(LABEL_KEY)] = y

    return outputs
""")

with open(TRANSFORM_MODULE, "w", encoding="utf-8") as f:
    f.write(MODULE_CODE)

print("✅ preprocessing_clean.py sudah dibuat.")