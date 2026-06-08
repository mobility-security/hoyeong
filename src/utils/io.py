import json
import numpy as np


def save_dataset(X: np.ndarray, y: np.ndarray, meta: dict, path: str) -> None:
    """데이터셋 저장. X: float32 (N,3,H,W) 범위 [0,1] / y: int64 (N,) 값 {0..5}"""
    np.savez(path, X=X, y=y, meta=json.dumps(meta))


def load_dataset(path: str):
    data = np.load(path, allow_pickle=True)
    X = data['X']
    y = data['y']

    # meta 키 이름이 파일마다 다를 수 있음 (stub: 'meta', 실제: 'meta_json')
    if 'meta_json' in data:
        meta = json.loads(bytes(data['meta_json']).decode('utf-8'))
    elif 'meta' in data:
        meta = json.loads(str(data['meta']))
    else:
        meta = {}

    # pcap_id가 있으면 meta에 포함 (시간적 누수 방지 분할에 사용됨)
    if 'pcap_id' in data:
        meta['pcap_id'] = data['pcap_id'].tolist()

    validate_schema(X, y)
    return X, y, meta


def validate_schema(X: np.ndarray, y: np.ndarray) -> None:
    assert X.dtype == np.float32,         f'X dtype {X.dtype} != float32'
    assert X.ndim == 4,                   f'X ndim {X.ndim} != 4'
    assert X.shape[1] == 3,               f'X channels {X.shape[1]} != 3'
    assert X.min() >= 0.0,                f'X min {X.min()} < 0'
    assert X.max() <= 1.0,                f'X max {X.max()} > 1'
    assert y.dtype == np.int64,           f'y dtype {y.dtype} != int64'
    assert y.ndim == 1,                   f'y ndim {y.ndim} != 1'
    assert set(y).issubset({0, 1, 2, 3, 4, 5}), f'y값 범위 위반: {set(y)}'
    assert len(X) == len(y),              f'X/y 길이 불일치'
    assert np.isfinite(X).all(),          'X에 nan/inf 존재'
