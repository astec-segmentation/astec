
import numpy as np
import h5py

from astec.components.spatial_image import SpatialImage

def read_h5(path):
    """Read an hdf5 file

    :Parameters:
     - `filename` (str) - name of the file to read
    """
    hf = h5py.File(path.replace('\\', ''), 'r')
    data = hf.get('Data')
    im_out = np.zeros(data.shape, dtype=data.dtype)
    data.read_direct(im_out)
    hf.close()
    return SpatialImage(im_out.transpose(2, 1, 0))
