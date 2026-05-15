try:
    from preprocess.generate_hypergraph import (
        generate_hypergraph_from_file,
        generate_hyperedge_stat,
        generate_traj2traj_data,
        generate_ci2traj_pyg_data,
        merge_traj2traj_data,
        filter_chunk
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"torch_sparse", "torch_geometric"}:
        raise

    def _missing_hypergraph_dependency(*args, **kwargs):
        raise ModuleNotFoundError(
            "Hypergraph preprocessing requires optional PyG dependencies "
            "torch_sparse and torch_geometric."
        ) from exc

    generate_hypergraph_from_file = _missing_hypergraph_dependency
    generate_hyperedge_stat = _missing_hypergraph_dependency
    generate_traj2traj_data = _missing_hypergraph_dependency
    generate_ci2traj_pyg_data = _missing_hypergraph_dependency
    merge_traj2traj_data = _missing_hypergraph_dependency
    filter_chunk = _missing_hypergraph_dependency
from preprocess.preprocess_fn import (
    remove_unseen_user_poi,
    id_encode,
    ignore_first,
    only_keep_last
)
from preprocess.file_reader import (
    FileReaderBase,
    FileReader
)
from preprocess.preprocess_main import (
    preprocess
)

__all__ = [
    "FileReaderBase",
    "FileReader",
    "generate_hypergraph_from_file",
    "generate_hyperedge_stat",
    "generate_traj2traj_data",
    "generate_ci2traj_pyg_data",
    "merge_traj2traj_data",
    "filter_chunk",
    "remove_unseen_user_poi",
    "id_encode",
    "ignore_first",
    "only_keep_last",
    "preprocess"
]
