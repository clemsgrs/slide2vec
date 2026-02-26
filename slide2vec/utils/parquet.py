import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def require_pyarrow():
    return pa, pq, ds
