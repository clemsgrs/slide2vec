"""Behavior and work bounds for incremental process-list reconciliation."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from slide2vec.runtime.persistence import update_process_list_after_embedding


def test_process_list_normalizes_sample_ids_once(tmp_path, monkeypatch):
    path = tmp_path / 'process_list.csv'
    path.write_text('sample_id,feature_status\na,tbp\nb,tbp\nc,tbp\nuntouched,tbp\n')
    original = pd.Series.astype
    conversions = []

    def counted(series, *args, **kwargs):
        if series.name == 'sample_id':
            conversions.append(len(series))
        return original(series, *args, **kwargs)

    monkeypatch.setattr(pd.Series, 'astype', counted)
    update_process_list_after_embedding(
        path, successful_slides=[SimpleNamespace(sample_id=s) for s in ['a', 'b', 'c', 'a']],
        persist_tile_embeddings=True, persist_hierarchical_embeddings=False,
        include_slide_embeddings=False, encoder_name='encoder', output_variant=None,
        tile_artifacts=[SimpleNamespace(sample_id='a', annotation=None, path=Path('/features/a.pt'))],
        hierarchical_artifacts=[], slide_artifacts=[],
    )
    result = pd.read_csv(path).fillna('')
    assert result[['sample_id', 'feature_status', 'feature_path']].values.tolist() == [
        ['a', 'success', '/features/a.pt'], ['b', 'error', ''], ['c', 'error', ''], ['untouched', 'tbp', ''],
    ]
    assert len(conversions) <= 1


def test_process_list_keeps_unfinished_annotation_rows_unchanged(tmp_path):
    path = tmp_path / 'process_list.csv'
    path.write_text('sample_id,annotation,feature_status,feature_path,aggregation_status\n'
                    'a,tumor,tbp,,tbp\na,stroma,tbp,,tbp\na,tissue,tbp,,tbp\n'
                    'a,tumor,tbp,,tbp\nb,tumor,tbp,,tbp\n')
    update_process_list_after_embedding(
        path, successful_slides=[SimpleNamespace(sample_id='a'), SimpleNamespace(sample_id='a')],
        persist_tile_embeddings=False, persist_hierarchical_embeddings=False,
        include_slide_embeddings=True, encoder_name='encoder', output_variant='default',
        tile_artifacts=[], hierarchical_artifacts=[],
        slide_artifacts=[SimpleNamespace(sample_id='a', annotation='tumor', path=Path('/features/a.pt')),
                         SimpleNamespace(sample_id='a', annotation=None, path=Path('/features/flat.pt'))],
    )
    result = pd.read_csv(path).fillna('')
    assert result[['feature_status', 'feature_path', 'aggregation_status']].values.tolist() == [
        ['success', '/features/a.pt', 'success'], ['tbp', '', 'tbp'],
        ['success', '/features/flat.pt', 'success'], ['success', '/features/a.pt', 'success'],
        ['tbp', '', 'tbp'],
    ]
