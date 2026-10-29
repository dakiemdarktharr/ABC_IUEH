"""Draft V3-backed selected-dataset window traversal.

This low-level helper accepts an already open, independently authenticated HDF5
file. It DOES NOT open a source, establish authorization, or enforce the target
lock. The production source remains disabled until a reviewed root connects
these duties. Only generated in-memory fixtures have exercised this helper.
"""
from __future__ import annotations

from . import window_reader


def iter_selected_open_file(source, source_file, *, phase, check_guard, deadline):
    """Traverse fixed validation metadata and read exact context or endpoint slices.

    Caller must hold the reviewed same-FD source-hash capability, pre-outcome
    plan approval and (for endpoints) the independently verified prediction
    lock. Passing a source-like object is not authentication.
    """
    if phase not in ('contexts','endpoints'): raise ValueError('invalid validation read phase')
    users=tuple(source._users); rows=tuple(source._rows); binding=source._binding
    loader=source._loader; reader=source._reader
    if len(users)!=5 or len(set(users))!=5 or set(users) != set(binding.ownership[source.fold_id]['outer_validation']):
        raise PermissionError('validation cohort is not the fixed five users')
    if set(users) & set(binding.ownership[source.fold_id]['outer_fit']):
        raise PermissionError('validation and fit roles overlap')
    if any(user not in binding.parent.expected_train.user_sessions for user in users):
        raise PermissionError('validation cohort is outside train15')
    audit=loader.ReadAudit(int(source_file.id.id), len(users), source_hash_verified=True)
    loader._validate_source_root_with_legacy_bridge(reader,source_file,binding,source_hash_verified=True)
    addresses={}
    def child(parent,name,path,group):
        obj=loader._get_selected_child(parent,name,expected_path=path,source_file=source_file,
                                       fit_users=set(users),audit=audit,group=group)
        reader._register_unique(obj,path,addresses)
        return obj
    root=child(source_file,'users','/users',True)
    by_key={(row.user_id,row.session_id,row.segment_id):row for row in rows}
    if len(by_key)!=len(rows): raise ValueError('duplicate selected validation segment')
    selected={}
    for user in users:
        user_path=f'/users/{user}'; user_group=child(root,user,user_path,True)
        if set(user_group.keys()) != {'sessions'}: raise ValueError('invalid selected user hierarchy')
        sessions=child(user_group,'sessions',f'{user_path}/sessions',True)
        expected=binding.parent.expected_train.user_sessions[user]
        if set(sessions.keys()) != set(expected): raise ValueError('selected sessions differ from metadata')
        for session in sorted(expected):
            path=f'{user_path}/sessions/{session}'; sess=child(sessions,session,path,True)
            if set(sess.keys()) != {'segments'}: raise ValueError('invalid selected session hierarchy')
            segments=child(sess,'segments',f'{path}/segments',True)
            if set(segments.keys()) != set(expected[session]): raise ValueError('selected segments differ from metadata')
            session_samples=0
            for segment in sorted(expected[session]):
                seg_path=f'{path}/segments/{segment}'; seg=child(segments,segment,seg_path,True)
                if set(seg.keys()) != {'timestamps','quaternion'}: raise ValueError('invalid selected segment hierarchy')
                timestamps=child(seg,'timestamps',f'{seg_path}/timestamps',False)
                quaternion=child(seg,'quaternion',f'{seg_path}/quaternion',False)
                n_t=reader._dataset_metadata(timestamps,dtype='float64',ndim=1)
                n_q=reader._dataset_metadata(quaternion,dtype='float32',ndim=2)
                row=by_key.get((user,session,segment))
                if (row is None or row.fold_id != source.fold_id or n_q[1:] != (4,) or
                    n_t != (n_q[0],) or n_q[0] < 1000 or row.count != n_q[0]-419):
                    raise ValueError('selected dataset differs from fixed origin metadata')
                selected[(user,session,segment)]=quaternion
                session_samples+=n_q[0]
            if session_samples != binding.parent.expected_train.session_windows[(user,session)] + 419*len(expected[session]):
                raise ValueError('selected session sample count differs from metadata')
    if set(selected) != set(by_key): raise ValueError('selected rows are incomplete')
    allowed=frozenset(dataset.name for dataset in selected.values())
    dataset_audit=loader._DatasetAudit(int(source_file.id.id),allowed,6400,deadline,audit)
    counts=[0]*5
    for row in rows:
        user_index=users.index(row.user_id); dataset=selected[(row.user_id,row.session_id,row.segment_id)]
        def read_rows(start,stop):
            check_guard()
            chunk=loader._read_dataset_rows(dataset,start=start,stop=stop,audit=dataset_audit)
            return chunk.astype('<f4',copy=False).tobytes(order='C')
        options={'user_index':user_index,'global_offset':int(row.offset_int64),
                 'origin_count':int(row.count),'sample_count':int(dataset.shape[0]),
                 'read_rows':read_rows,'check_guard':check_guard}
        stream=(window_reader.iter_segment_contexts if phase == 'contexts' else window_reader.iter_segment_endpoints)(**options)
        for item in stream:
            counts[user_index]+=1
            yield item
    if audit.denied_reads or tuple(counts) != tuple(source.expected_origin_counts):
        raise PermissionError('selected source read differs from full fixed scope')
