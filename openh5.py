def traverse_datasets(hdf_file):

    """Traverse all datasets across all groups in HDF5 file."""

    import h5py

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, 'r') as f:
        for (path, dset) in h5py_dataset_iterator(f):
            print(path, dset)

    return None

traverse_datasets('file.h5')
# /DataSet1 <HDF5 dataset "DataSet1": shape (655559, 260), type "<f4">
# /DataSet2 <HDF5 dataset "DataSet2": shape (22076, 10000), type "<f4">
# /index <HDF5 dataset "index": shape (677635,), type "|V384">

with h5pyFile('file.h5', 'r') as f:
    arr = f['/DataSet1'][:]  # read entire dataset into memory