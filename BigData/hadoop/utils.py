import h5py
import numpy


def read_hdf5(hdf5file, show, dataset_list=("cep", "vad")):
    """

    :param h5f: HDF5 filename to read from
    :param show: identifier of the show to read
    :param dataset_list: list of datasets to read and concatenate
    :return:
    """
    with h5py.File(hdf5file, 'r') as h5f:
	    if show not in h5f:
	        raise Exception('show {} is not in the HDF5 file'.format(show))

	    feat = []
	    if "energy" in dataset_list:
	        if "/".join((show, "energy")) in h5f:
	            feat.append(h5f["/".join((show, "energy"))][:, numpy.newaxis])
	        else:
	            raise Exception('energy is not in the HDF5 file')
	    if "cep" in dataset_list:
	        if "/".join((show, "cep")) in h5f:
	            feat.append(h5f["/".join((show, "cep"))][()])
	        else:
	            raise Exception('cep is not in the HDF5 file')
	    if "fb" in dataset_list:
	        if "/".join((show, "fb")) in h5f:
	            feat.append(h5f["/".join((show, "fb"))][()])
	        else:
	            raise Exception('fb is not in the HDF5 file')
	    if "bnf" in dataset_list:
	        if "/".join((show, "bnf")) in h5f:
	            feat.append(h5f["/".join((show, "bnf"))][()])
	        else:
	            raise Exception('bnf is not in the HDF5 file')
	    feat = numpy.hstack(feat)

	    label = None
	    if "vad" in dataset_list:
	        if "/".join((show, "vad")) in h5f:
	            label = h5f.get("/".join((show, "vad")))[()].astype('bool').squeeze()
	        else:
	            warnings.warn("Warning...........no VAD in this HDF5 file")
	            label = numpy.ones(feat.shape[0], dtype='bool')

	    return feat.astype(numpy.float32), label