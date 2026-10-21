import os
import platform
if platform.system() == 'Windows':
    import winfcntl as fcntl
else:
    import fcntl
import pickle
import numpy as np



def _empty_samples_dict(im_ids, nsamples, lensing_columns) :
    return {imID: {col: np.full(nsamples, np.nan) for col in lensing_columns} for imID in im_ids}


def _sample_is_computed(samples_dict, sample_index, im_ids, lensing_columns) :
    for im_id in im_ids :
        for col in lensing_columns :
            if np.isnan(samples_dict[im_id][col][sample_index]) :
                return False
    return True


def _validate_samples_dict(samples_dict, nsamples, lensing_columns, im_ids) :
    if set(samples_dict.keys()) != set(im_ids) :
        raise ValueError('Existing samples dictionary image IDs do not match the current catalog.')
    for im_id in im_ids :
        for col in lensing_columns :
            if col not in samples_dict[im_id] :
                raise ValueError(f'Existing samples dictionary is missing column {col!r} for image {im_id!r}.')
            if len(samples_dict[im_id][col]) != nsamples :
                raise ValueError(
                    f'Existing samples dictionary length ({len(samples_dict[im_id][col])}) '
                    f'does not match nsamples ({nsamples}).'
                )


def _atomic_pickle_dump(obj, path) :
    tmp_path = f"{path}{'.tmp'}"
    with open(tmp_path, 'wb') as f :
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def _load_or_init_samples_dict(samples_dict_path, nsamples, lensing_columns, im_ids, start_sample_index, stop_sample_index, recompute) :
    lock_path = f"{samples_dict_path}{'.lock'}"
    with open(lock_path, 'w') as lock_file :
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try :
            if os.path.exists(samples_dict_path) :
                with open(samples_dict_path, 'rb') as f :
                    samples_dict = pickle.load(f)
                _validate_samples_dict(samples_dict, nsamples, lensing_columns, im_ids)
            else :
                samples_dict = _empty_samples_dict(im_ids, nsamples, lensing_columns)
                _atomic_pickle_dump(samples_dict, samples_dict_path)

            if recompute :
                for im_id in im_ids :
                    for col in lensing_columns :
                        samples_dict[im_id][col][start_sample_index:stop_sample_index] = np.nan
                _atomic_pickle_dump(samples_dict, samples_dict_path)
        finally :
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return samples_dict


def _merge_and_dump_samples_dict(samples_dict_path, local_dict, start_sample_index, stop_sample_index, nsamples, lensing_columns, im_ids) :
    lock_path = f"{samples_dict_path}{'.lock'}"
    with open(lock_path, 'w') as lock_file :
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try :
            if os.path.exists(samples_dict_path) :
                with open(samples_dict_path, 'rb') as f :
                    merged = pickle.load(f)
                _validate_samples_dict(merged, nsamples, lensing_columns, im_ids)
            else :
                merged = _empty_samples_dict(im_ids, nsamples, lensing_columns)

            for im_id in im_ids :
                for col in lensing_columns :
                    merged[im_id][col][start_sample_index:stop_sample_index] = local_dict[im_id][col][start_sample_index:stop_sample_index]

            _atomic_pickle_dump(merged, samples_dict_path)

            for im_id in im_ids :
                for col in lensing_columns :
                    local_dict[im_id][col][:] = merged[im_id][col]
        finally :
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return local_dict
