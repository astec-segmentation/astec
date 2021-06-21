import os
import imp
import shutil
import sys
import time

from astec.utils import common
from astec.components.spatial_image import SpatialImage
from astec.io.image import imread
from astec.wrapping import cpp_wrapping

#
#
#
#
#

monitoring = common.Monitoring()


########################################################################################
#
# classes
# - computation parameters
#
########################################################################################


class IntraRegParameters(common.PrefixedParameter):

    ############################################################
    #
    # initialisation
    #
    ############################################################

    def __init__(self):

        common.PrefixedParameter.__init__(self, prefix=['intra_registration_'])

        if "doc" not in self.__dict__:
            self.doc = {}

        #
        # Co-registration parameters
        #

        self.registration = common.RegistrationParameters(prefix=self._prefix)
        self.registration.transformation_type = 'rigid'

        #
        # intra-sequence transformation parameters
        # reference, ie image to be still (up to a translation)
        # while composing transformation
        # 'reference_transformation_file' is the transformation
        # that will be resampled the whole series afterwards
        # (defined for the reference image)
        #

        doc = "\t defines the still image after transformation compositions\n"
        doc += "\t it will only translated, except if 'reference_transformation_file'\n"
        doc += "\t or 'reference_transformation_angles' are given\n"
        doc += "\t \n"
        self.doc['reference_index'] = doc
        self.reference_index = None

        doc = "\t resampling transformation to be applied to the reference\n"
        doc += "\t image (and to the whole serie) after transformation \n"
        doc += "\t compositions.\n"
        doc += "\t \n"
        self.doc['reference_transformation_file'] = doc
        self.reference_transformation_file = None

        doc = "\t list of rotations wrt the X, Y,or Z axis that defines the\n"
        doc += "\t resampling transformation. \n"
        doc += "\t syntax: 'X 30 Y 50' means a rotation of 30 degree around\n"
        doc += "\t X followed by a rotation of 50 around Y\n"
        doc += "\t beware: rotation composition depends on the order, so\n"
        doc += "\t 'X 30 Y 50' is not equivalent to 'Y 50 X 30'\n"
        self.doc['reference_transformation_angles'] = doc
        self.reference_transformation_angles = None

        #
        # intra-sequence transformation parameters
        # input template, ie how to define the useful information to be kept
        #
        doc = "\t Possible values are 'FUSION', 'SEGMENTATION', or 'POST-SEGMENTATION'\n"
        doc += "\t The template is built so that the useful information of\n"
        doc += "\t all resampled images fits into it. Useful information\n"
        doc += "\t can be issued from either the fused sequence, the segmentation\n"
        doc += "\t sequence or the post-segmentation sequence. \n"
        self.doc['template_type'] = doc
        self.template_type = "FUSION"

        doc = "\t Giving a threshold with the 'template_type', only points\n"
        doc += "\t above the threshold are considered to be included in the\n"
        doc += "\t template after resampling, this allows to reduce the template.\n"
        doc += "\t According the background value is either 0 or 1 in both the\n"
        doc += "\t segmentation and the post-segmentation sequences, setting\n"
        doc += "\t this threshold to 2 for these sequences allows to keep the\n"
        doc += "\t entire embryo in the resampled/reconstructed sequence.\n"
        self.doc['template_threshold'] = doc
        self.template_threshold = None

        doc = "\t In addition, a margin can be given for a more comfortable\n"
        doc += "\t visualization. By default, it is 0 when only fusion\n"
        doc += "\t images are used, and 10 if either segmentation or\n"
        doc += "\t post-segmentation images are also used\n"
        self.doc['margin'] = doc
        self.margin = None

        #
        # output template
        #
        doc = "\t gives the resulting (isotropic) voxel size (as the \n"
        doc += "\t 'target_resolution' gives the voxel size of the fused images).\n"
        doc += "\t However, for visualization purposes, it may be indicated to\n"
        doc += "\t have a larger voxel size (hence the 0.6 instead of 0.3)\n"
        self.doc['resolution'] = doc
        self.resolution = 0.3

        #
        # force rebuilding of template and of transformations versus a reference
        # useful when transformations have already been computed for fused image as template
        # they can re-used for segmentation images as template
        #
        doc = "\t Possible values are True or False\n"
        doc += "\t if True, force to recompute the template as well as the \n"
        doc += "\t transformations from the co-registrations (that are not\n"
        doc += "\t re-computed). It is useful when a first intra-registration\n"
        doc += "\t has been done with only the fusion images: a second\n"
        doc += "\t  intra-registration with the segmentation images as template \n"
        doc += "\t can be done without recomputing the co-registration\n"
        self.doc['rebuild_template'] = doc
        self.rebuild_template = False

        #
        # resampling parameters
        #
        doc = "\t Sigma to smooth (post-)segmentation images when resampling\n"
        self.doc['sigma_segmentation_images'] = doc
        self.sigma_segmentation_images = 1.0

        doc = "\t Possible values are True or False\n"
        self.doc['resample_fusion_images'] = doc
        self.resample_fusion_images = True
        doc = "\t Possible values are True or False \n"
        self.doc['resample_segmentation_images'] = doc
        self.resample_segmentation_images = False
        doc = "\t Possible values are True or False \n"
        self.doc['resample_post_segmentation_images'] = doc
        self.resample_post_segmentation_images = False

        doc = "\t Possible values are True or False\n"
        doc += "\t To build 2D+t movies from the resampled fusion images.\n"
        self.doc['movie_fusion_images'] = doc
        self.movie_fusion_images = True

        doc = "\t Possible values are True or False\n"
        doc += "\t To build 2D+t movies from the resampled segmentation .\n"
        doc += "\t images\n"
        self.doc['movie_segmentation_images'] = doc
        self.movie_segmentation_images = False

        doc = "\t Possible values are True or False\n"
        doc += "\t To build 2D+t movies from the resampled post-segmentation.\n"
        doc += "\t images\n"
        self.doc['movie_post_segmentation_images'] = doc
        self.movie_post_segmentation_images = False

        doc = "\t List of XY-sections used to build the 2D+t movies\n"
        doc += "\t eg 'xy_movie_fusion_images = [100, 200]'\n"
        self.doc['xy_movie_fusion_images'] = doc
        self.xy_movie_fusion_images = []

        doc = "\t List of XZ-sections used to build the 2D+t movies\n"
        self.doc['xz_movie_fusion_images'] = doc
        self.xz_movie_fusion_images = []

        doc = "\t List of YZ-sections used to build the 2D+t movies\n"
        self.doc['yz_movie_fusion_images'] = doc
        self.yz_movie_fusion_images = []

        doc = "\t List of XY-sections used to build the 2D+t movies\n"
        doc += "\t eg 'xy_movie_segmentation_images = [100, 200]'\n"
        self.doc['xy_movie_segmentation_images'] = doc
        self.xy_movie_segmentation_images = []

        doc = "\t List of XZ-sections used to build the 2D+t movies\n"
        self.doc['xz_movie_segmentation_images'] = doc
        self.xz_movie_segmentation_images = []

        doc = "\t List of YZ-sections used to build the 2D+t movies\n"
        self.doc['yz_movie_segmentation_images'] = doc
        self.yz_movie_segmentation_images = []

        doc = "\t List of XY-sections used to build the 2D+t movies\n"
        doc += "\t eg 'xy_movie_segmentation_images = [100, 200]'\n"
        self.doc['xy_movie_post_segmentation_images'] = doc
        self.xy_movie_post_segmentation_images = []

        doc = "\t List of XZ-sections used to build the 2D+t movies\n"
        self.doc['xz_movie_post_segmentation_images'] = doc
        self.xz_movie_post_segmentation_images = []

        doc = "\t List of YZ-sections used to build the 2D+t movies\n"
        self.doc['yz_movie_post_segmentation_images'] = doc
        self.yz_movie_post_segmentation_images = []

        doc = "\t Possible values are True or False\n"
        doc += "\t build a maximum image from the resampled series\n"
        doc += "\t it may be useful to define a cropping valid area\n"
        doc += "\t for the whole sequence\n"
        self.doc['maximum_fusion_images'] = doc
        self.maximum_fusion_images = False

        doc = "\t Possible values are True or False\n"
        doc += "\t build a maximum image from the resampled series\n"
        doc += "\t it may be useful to define a cropping valid area\n"
        doc += "\t for the whole sequence\n"
        self.doc['maximum_segmentation_images'] = doc
        self.maximum_segmentation_images = False

        doc = "\t Possible values are True or False\n"
        doc += "\t build a maximum image from the resampled series\n"
        doc += "\t it may be useful to define a cropping valid area\n"
        doc += "\t for the whole sequence\n"
        self.doc['maximum_post_segmentation_images'] = doc
        self.maximum_post_segmentation_images = False

    ############################################################
    #
    # print / write
    #
    ############################################################

    def print_parameters(self):
        print("")
        print('#')
        print('# IntraRegParameters')
        print('#')
        print("")

        common.PrefixedParameter.print_parameters(self)

        #
        # Co-registration parameters
        #

        self.registration.print_parameters()

        #
        # intra-sequence transformation parameters
        #

        self.varprint('reference_index', self.reference_index, self.doc['reference_index'])
        self.varprint('reference_transformation_file', self.reference_transformation_file,
                      self.doc['reference_transformation_file'])
        self.varprint('reference_transformation_angles', self.reference_transformation_angles,
                      self.doc['reference_transformation_angles'])

        self.varprint('template_type', self.template_type, self.doc['template_type'])
        self.varprint('template_threshold', self.template_threshold, self.doc['template_threshold'])
        self.varprint('margin', self.margin, self.doc['margin'])

        self.varprint('resolution', self.resolution, self.doc['resolution'])

        self.varprint('rebuild_template', self.rebuild_template, self.doc['rebuild_template'])

        #
        # resampling parameters
        #

        self.varprint('sigma_segmentation_images', self.sigma_segmentation_images,
                      self.doc['sigma_segmentation_images'])
        self.varprint('resample_fusion_images', self.resample_fusion_images, self.doc['resample_fusion_images'])
        self.varprint('resample_segmentation_images', self.resample_segmentation_images,
                      self.doc['resample_segmentation_images'])
        self.varprint('resample_post_segmentation_images', self.resample_post_segmentation_images,
                      self.doc['resample_post_segmentation_images'])

        #
        # movie parameters
        #

        self.varprint('movie_fusion_images', self.movie_fusion_images, self.doc['movie_fusion_images'])
        self.varprint('movie_segmentation_images', self.movie_segmentation_images,
                      self.doc['movie_segmentation_images'])
        self.varprint('movie_post_segmentation_images', self.movie_post_segmentation_images,
                      self.doc['movie_post_segmentation_images'])

        self.varprint('xy_movie_fusion_images', self.xy_movie_fusion_images, self.doc['xy_movie_fusion_images'])
        self.varprint('xz_movie_fusion_images', self.xz_movie_fusion_images, self.doc['xz_movie_fusion_images'])
        self.varprint('yz_movie_fusion_images', self.yz_movie_fusion_images, self.doc['yz_movie_fusion_images'])

        self.varprint('xy_movie_segmentation_images', self.xy_movie_segmentation_images,
                      self.doc['xy_movie_segmentation_images'])
        self.varprint('xz_movie_segmentation_images', self.xz_movie_segmentation_images,
                      self.doc['xz_movie_segmentation_images'])
        self.varprint('yz_movie_segmentation_images', self.yz_movie_segmentation_images,
                      self.doc['yz_movie_segmentation_images'])

        self.varprint('xy_movie_post_segmentation_images', self.xy_movie_post_segmentation_images,
                      self.doc['xy_movie_post_segmentation_images'])
        self.varprint('xz_movie_post_segmentation_images', self.xz_movie_post_segmentation_images,
                      self.doc['xz_movie_post_segmentation_images'])
        self.varprint('yz_movie_post_segmentation_images', self.yz_movie_post_segmentation_images,
                      self.doc['yz_movie_post_segmentation_images'])

        self.varprint('maximum_fusion_images', self.maximum_fusion_images, self.doc['maximum_fusion_images'])
        self.varprint('maximum_segmentation_images', self.maximum_segmentation_images,
                      self.doc['maximum_segmentation_images'])
        self.varprint('maximum_post_segmentation_images', self.maximum_post_segmentation_images,
                      self.doc['maximum_post_segmentation_images'])

        print("")
        return

    def write_parameters_in_file(self, logfile):
        logfile.write("\n")
        logfile.write('#\n')
        logfile.write('# IntraRegParameters\n')
        logfile.write('#\n')
        logfile.write("\n")

        common.PrefixedParameter.write_parameters_in_file(self, logfile)

        #
        # Co-registration parameters
        #

        self.registration.write_parameters_in_file(logfile)

        #
        # intra-sequence transformation parameters
        #

        self.varwrite(logfile, 'reference_index', self.reference_index, self.doc['reference_index'])
        self.varwrite(logfile, 'reference_transformation_file', self.reference_transformation_file,
                      self.doc['reference_transformation_file'])
        self.varwrite(logfile, 'reference_transformation_angles', self.reference_transformation_angles,
                      self.doc['reference_transformation_angles'])

        self.varwrite(logfile, 'template_type', self.template_type, self.doc['template_type'])
        self.varwrite(logfile, 'template_threshold', self.template_threshold, self.doc['template_threshold'])
        self.varwrite(logfile, 'margin', self.margin, self.doc['margin'])

        self.varwrite(logfile, 'resolution', self.resolution, self.doc['resolution'])

        self.varwrite(logfile, 'rebuild_template', self.rebuild_template, self.doc['rebuild_template'])

        #
        # resampling parameters
        #

        self.varwrite(logfile, 'sigma_segmentation_images', self.sigma_segmentation_images,
                      self.doc['sigma_segmentation_images'])
        self.varwrite(logfile, 'resample_fusion_images', self.resample_fusion_images,
                      self.doc['resample_fusion_images'])
        self.varwrite(logfile, 'resample_segmentation_images', self.resample_segmentation_images,
                      self.doc['resample_segmentation_images'])
        self.varwrite(logfile, 'resample_post_segmentation_images', self.resample_post_segmentation_images,
                      self.doc['resample_post_segmentation_images'])

        #
        # movie parameters
        #

        self.varwrite(logfile, 'movie_fusion_images', self.movie_fusion_images, self.doc['movie_fusion_images'])
        self.varwrite(logfile, 'movie_segmentation_images', self.movie_segmentation_images,
                      self.doc['movie_segmentation_images'])
        self.varwrite(logfile, 'movie_post_segmentation_images', self.movie_post_segmentation_images,
                      self.doc['movie_post_segmentation_images'])

        self.varwrite(logfile, 'xy_movie_fusion_images', self.xy_movie_fusion_images,
                      self.doc['xy_movie_fusion_images'])
        self.varwrite(logfile, 'xz_movie_fusion_images', self.xz_movie_fusion_images,
                      self.doc['xz_movie_fusion_images'])
        self.varwrite(logfile, 'yz_movie_fusion_images', self.yz_movie_fusion_images,
                      self.doc['yz_movie_fusion_images'])

        self.varwrite(logfile, 'xy_movie_segmentation_images', self.xy_movie_segmentation_images,
                      self.doc['xy_movie_segmentation_images'])
        self.varwrite(logfile, 'xz_movie_segmentation_images', self.xz_movie_segmentation_images,
                      self.doc['xz_movie_segmentation_images'])
        self.varwrite(logfile, 'yz_movie_segmentation_images', self.yz_movie_segmentation_images,
                      self.doc['yz_movie_segmentation_images'])

        self.varwrite(logfile, 'xy_movie_post_segmentation_images', self.xy_movie_post_segmentation_images,
                      self.doc['xy_movie_post_segmentation_images'])
        self.varwrite(logfile, 'xz_movie_post_segmentation_images', self.xz_movie_post_segmentation_images,
                      self.doc['xz_movie_post_segmentation_images'])
        self.varwrite(logfile, 'yz_movie_post_segmentation_images', self.yz_movie_post_segmentation_images,
                      self.doc['yz_movie_post_segmentation_images'])

        self.varwrite(logfile, 'maximum_fusion_images', self.maximum_fusion_images, self.doc['maximum_fusion_images'])
        self.varwrite(logfile, 'maximum_segmentation_images', self.maximum_segmentation_images,
                      self.doc['maximum_segmentation_images'])
        self.varwrite(logfile, 'maximum_post_segmentation_images', self.maximum_post_segmentation_images,
                      self.doc['maximum_post_segmentation_images'])

        logfile.write("\n")
        return

    def write_parameters(self, log_filename=None):
        if log_filename is not None:
            local_log_filename = log_filename
        else:
            local_log_filename = monitoring.log_filename
        if local_log_filename is not None:
            with open(local_log_filename, 'a') as logfile:
                self.write_parameters_in_file(logfile)
        return

    ############################################################
    #
    # update
    #
    ############################################################

    def update_from_args(self, args):
        self.reference_transformation_file = args.reference_transformation_file
        self.reference_transformation_angles = args.reference_transformation_angles

    def update_from_parameters(self, parameters):

        margin_is_updated = False

        #
        # co-registration parameters
        #

        self.registration.update_from_parameters(parameters)

        #
        # intra-sequence transformation parameters
        #
        self.reference_index = self.read_parameter(parameters, 'reference_index', self.reference_index)
        self.reference_transformation_file = self.read_parameter(parameters, 'reference_transformation_file', self.reference_transformation_file)
        self.reference_transformation_angles = self.read_parameter(parameters, 'reference_transformation_angles', self.reference_transformation_angles)

        self.template_type = self.read_parameter(parameters, 'template_type', self.template_type)
        self.template_threshold = self.read_parameter(parameters, 'template_threshold', self.template_threshold)
        old_margin = self.margin
        self.margin = self.read_parameter(parameters, 'margin', self.margin)
        if self.margin != old_margin:
            margin_is_updated = True

        self.resolution = self.read_parameter(parameters, 'resolution', self.resolution)

        self.rebuild_template = self.read_parameter(parameters, 'rebuild_template', self.rebuild_template)

        #
        # resampling parameters
        #

        self.sigma_segmentation_images = self.read_parameter(parameters, 'sigma_segmentation_images', self.sigma_segmentation_images)

        self.resample_fusion_images = self.read_parameter(parameters, 'resample_fusion_images', self.resample_fusion_images)
        self.resample_segmentation_images = self.read_parameter(parameters, 'resample_segmentation_images', self.resample_segmentation_images)
        self.resample_post_segmentation_images = self.read_parameter(parameters, 'resample_post_segmentation_images', self.resample_post_segmentation_images)

        self.movie_fusion_images = self.read_parameter(parameters, 'movie_fusion_images', self.movie_fusion_images)
        self.movie_segmentation_images = self.read_parameter(parameters, 'movie_segmentation_images', self.movie_segmentation_images)
        self.movie_post_segmentation_images = self.read_parameter(parameters, 'movie_post_segmentation_images', self.movie_post_segmentation_images)

        self.xy_movie_fusion_images = self.read_parameter(parameters, 'xy_movie_fusion_images', self.xy_movie_fusion_images)
        self.xz_movie_fusion_images = self.read_parameter(parameters, 'xz_movie_fusion_images', self.xz_movie_fusion_images)
        self.yz_movie_fusion_images = self.read_parameter(parameters, 'yz_movie_fusion_images', self.yz_movie_fusion_images)

        self.xy_movie_segmentation_images = self.read_parameter(parameters, 'xy_movie_segmentation_images', self.xy_movie_segmentation_images)
        self.xz_movie_segmentation_images = self.read_parameter(parameters, 'xz_movie_segmentation_images', self.xz_movie_segmentation_images)
        self.yz_movie_segmentation_images = self.read_parameter(parameters, 'yz_movie_segmentation_images', self.yz_movie_segmentation_images)

        self.xy_movie_post_segmentation_images = self.read_parameter(parameters, 'xy_movie_post_segmentation_images', self.xy_movie_post_segmentation_images)
        self.xz_movie_post_segmentation_images = self.read_parameter(parameters, 'xz_movie_post_segmentation_images', self.xz_movie_post_segmentation_images)
        self.yz_movie_post_segmentation_images = self.read_parameter(parameters, 'yz_movie_post_segmentation_images', self.yz_movie_post_segmentation_images)

        self.maximum_fusion_images = self.read_parameter(parameters, 'maximum_fusion_images', self.maximum_fusion_images)
        self.maximum_segmentation_images = self.read_parameter(parameters, 'maximum_segmentation_images', self.maximum_segmentation_images)
        self.maximum_post_segmentation_images = self.read_parameter(parameters, 'maximum_post_segmentation_images', self.maximum_post_segmentation_images)

        #
        # default for margin is none (ie no margin)
        # which is convenient for fusion images
        # however, when dealing with segmentation images as template, a little margin will be great
        # thus, define the default as 10 (if no specification from the user)
        #
        if hasattr(parameters, 'intra_registration_template_type'):
            if parameters.intra_registration_template_type.lower() == 'segmentation' \
                    or parameters.intra_registration_template_type.lower() == 'seg' \
                    or parameters.intra_registration_template_type.lower() == 'post-segmentation' \
                    or parameters.intra_registration_template_type.lower() == 'post_segmentation' \
                    or parameters.intra_registration_template_type.lower() == 'post':
                if margin_is_updated is False:
                    parameters.intra_registration_margin = 10
        if hasattr(parameters, 'template_type'):
            if parameters.template_type.lower() == 'segmentation' \
                    or parameters.template_type.lower() == 'seg' \
                    or parameters.template_type.lower() == 'post-segmentation' \
                    or parameters.template_type.lower() == 'post_segmentation' \
                    or parameters.template_type.lower() == 'post':
                if margin_is_updated is False:
                    parameters.intra_registration_margin = 10


    def update_from_parameter_file(self, parameter_file):
        if parameter_file is None:
            return
        if not os.path.isfile(parameter_file):
            print("Error: '" + parameter_file + "' is not a valid file. Exiting.")
            sys.exit(1)

        parameters = imp.load_source('*', parameter_file)
        self.update_from_parameters(parameters)



########################################################################################
#
# define paths and file names
#
########################################################################################


def _get_cotrsf_path(experiment):
    """

    :param experiment:
    :return:
    """
    cotrsf_path = os.path.join(experiment.get_embryo_path(), experiment.intrareg_dir.get_directory(), 'CO-TRSFS')
    return cotrsf_path


def _get_cotrsf_path_name(experiment, floating_index, reference_index):
    """

    :param experiment:
    :param floating_index:
    :param reference_index:
    :return:
    """
    flo = experiment.get_time_index(floating_index)
    ref = experiment.get_time_index(reference_index)
    co_trsf_name = experiment.get_embryo_name() + '_intrareg_flo' + str(flo) + '_ref' + str(ref) + '.trsf'
    return os.path.join(_get_cotrsf_path(experiment), co_trsf_name)


def _get_cotrsf_path_format(experiment):
    """

    :param experiment:
    :return:
    """
    form = experiment.get_time_format()
    co_trsf_format = experiment.get_embryo_name() + '_intrareg_flo' + form + '_ref' + form + '.trsf'
    return os.path.join(_get_cotrsf_path(experiment), co_trsf_format)


def _get_trsf_path(experiment):
    """

    :param experiment:
    :return:
    """
    firstindex = experiment.first_time_point + experiment.delay_time_point
    lastindex = experiment.last_time_point + experiment.delay_time_point
    trsf_dir = 'TRSFS' + "_t" + str(firstindex) + "-" + str(lastindex)
    trsf_path = os.path.join(experiment.get_embryo_path(), experiment.intrareg_dir.get_directory(), trsf_dir)
    return trsf_path


def _get_trsf_name(experiment, index):
    """

    :param experiment:
    :param index:
    :return:
    """
    ind = experiment.get_time_index(index)
    trsf_name = experiment.get_embryo_name() + '_intrareg_t' + str(ind) + '.trsf'
    return trsf_name


def _get_trsf_format(experiment):
    """

    :param experiment:
    :return:
    """
    form = experiment.get_time_format()
    trsf_format = experiment.get_embryo_name() + '_intrareg_t' + form + '.trsf'
    return trsf_format


def _get_template_path_name(experiment):
    """

    :param experiment:
    :return:
    """
    firstindex = experiment.first_time_point + experiment.delay_time_point
    lastindex = experiment.last_time_point + experiment.delay_time_point
    result_template = "template" + "_t" + str(firstindex) + "-" + str(lastindex) + "." + experiment.result_image_suffix
    return os.path.join(_get_trsf_path(experiment), result_template)


########################################################################################
#
# some internal procedures
#
########################################################################################

def _check_data(experiment, suffix=None):
    """
    Check whether all the images (from the first time point to the last one) exist
    :param experiment:
    :param suffix:
    :return:
    """

    proc = "_check_data"

    first_time_point = experiment.first_time_point + experiment.delay_time_point
    last_time_point = experiment.last_time_point + experiment.delay_time_point

    path_fusion = os.path.join(experiment.get_embryo_path(), experiment.fusion_dir.get_directory())

    for current_time in range(first_time_point + experiment.delta_time_point,
                              last_time_point + 1, experiment.delta_time_point):

        input_name = experiment.get_image_name(current_time, 'fuse')

        if suffix is None:
            input_image = common.find_file(path_fusion, input_name, file_type='image', callfrom=proc,
                                           local_monitoring=monitoring)

            if input_image is None:
                monitoring.to_log_and_console("    .. image '" + input_name + "' not found in '"
                                              + str(path_fusion) + "'", 2)
                return False

        else:
            input_image = input_name + '.' + suffix
            if not os.path.isfile(os.path.join(path_fusion, input_image)):
                monitoring.to_log_and_console("    .. image '" + input_image + "' not found in '"
                                              + str(path_fusion) + "'", 2)
                return False

    return True


########################################################################################
#
#
#
########################################################################################


def _coregistration_control(experiment, parameters):
    """
    Perform the co-registration of any couple of two successive images
    Resulting transformations are computed with fused image at t as floating image
    and used image at t+delta_t as reference image

    :param experiment:
    :param parameters:
    :return:
    """

    proc = "_coregistration_control"

    #    if _check_data(experiment) is False:
    #        monitoring.to_log_and_console(proc + ": error, some fused data are missing", 1)
    #        monitoring.to_log_and_console("\t Exiting")
    #        sys.exit(1)

    path_intrareg_cotrsf = _get_cotrsf_path(experiment)
    if not os.path.isdir(path_intrareg_cotrsf):
        os.makedirs(path_intrareg_cotrsf)

    first_time_point = experiment.first_time_point + experiment.delay_time_point
    last_time_point = experiment.last_time_point + experiment.delay_time_point

    path_fusion = experiment.fusion_dir.get_directory()

    #
    # loop on time
    #

    for reference_time in range(first_time_point + experiment.delta_time_point,
                                last_time_point + 1, experiment.delta_time_point):

        floating_time = reference_time - experiment.delta_time_point
        trsf_name = _get_cotrsf_path_name(experiment, floating_time, reference_time)

        if not os.path.isfile(trsf_name) or monitoring.forceResultsToBeBuilt is True:

            floating_prefix = experiment.fusion_dir.get_image_name(floating_time)
            floating_name = common.find_file(path_fusion, floating_prefix, file_type='image', callfrom=proc,
                                             local_monitoring=monitoring)
            if floating_name is None:
                monitoring.to_log_and_console(proc + ": error, image '" + str(floating_prefix) + "' was not found", 1)
                monitoring.to_log_and_console("\t Exiting")
                sys.exit(1)

            reference_prefix = experiment.fusion_dir.get_image_name(reference_time)
            reference_name = common.find_file(path_fusion, reference_prefix, file_type='image', callfrom=proc,
                                              local_monitoring=monitoring)
            if reference_name is None:
                monitoring.to_log_and_console(proc + ": error, image '" + str(reference_prefix) + "' was not found", 1)
                monitoring.to_log_and_console("\t Exiting")
                sys.exit(1)

            floating_image = os.path.join(path_fusion, floating_name)
            reference_image = os.path.join(path_fusion, reference_name)
            monitoring.to_log_and_console("       co-registering '" + floating_name + "'", 2)
            cpp_wrapping.linear_registration(reference_image, floating_image, None, trsf_name, None,
                                             py_hl=parameters.pyramid_highest_level,
                                             py_ll=parameters.pyramid_lowest_level,
                                             transformation_type=parameters.transformation_type,
                                             transformation_estimator=parameters.transformation_estimation_type,
                                             lts_fraction=parameters.lts_fraction,
                                             normalization=parameters.normalization,
                                             monitoring=monitoring)

    return


def _transformations_from_reference(experiment, parameters, temporary_dir):
    """
    Combine the transformations issued from the co-registration of pairs of successive images
    to get transformations from one given image
    :param experiment:
    :param parameters:
    :return:
    """

    if not os.path.isdir(temporary_dir):
        os.makedirs(temporary_dir)

    first_time_point = experiment.first_time_point + experiment.delay_time_point
    last_time_point = experiment.last_time_point + experiment.delay_time_point

    #
    #
    #
    build_trsf = False
    if monitoring.forceResultsToBeBuilt is True:
        build_trsf = True
    else:
        for i in range(first_time_point + experiment.delta_time_point, last_time_point + 1,
                       experiment.delta_time_point):
            trsf_name = _get_trsf_name(experiment, i)
            if not os.path.isfile(os.path.join(temporary_dir, trsf_name)):
                build_trsf = True
                break

    #
    #
    #
    if build_trsf is True:
        if parameters.reference_index is None:
            reference_index = first_time_point
        else:
            reference_index = parameters.reference_index

        format_input = _get_cotrsf_path_format(experiment)

        trsf_format = _get_trsf_format(experiment)
        format_output = os.path.join(temporary_dir, trsf_format)

        cpp_wrapping.multiple_trsfs(format_input, format_output, first_time_point, last_time_point,
                                    reference_index, trsf_type=parameters.registration.transformation_type,
                                    monitoring=monitoring)

    return


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _get_reference_transformation(parameters, temporary_dir):
    """

    :param parameters:
    :return:
    """

    proc = "_get_reference_transformation"

    if parameters.reference_transformation_file is not None:
        if os.path.isfile(parameters.reference_transformation_file):
            return parameters.reference_transformation_file

    if parameters.reference_transformation_angles is not None:
        if type(parameters.reference_transformation_angles) == str:
            s = parameters.reference_transformation_angles.split(' ')
            create_trsf_option = "-angle-unit degree"
            le = len(create_trsf_option)
            i = 0
            while i < len(s):
                if s[i].lower() == 'x':
                    if i+1 < len(s) and _is_float(s[i+1]):
                        create_trsf_option += ' -xrotation ' + s[i + 1]
                    else:
                        monitoring.to_log_and_console(proc + ": weird value for 'X rotation' -> " + str(s[i + 1]), 1)
                elif s[i].lower() == 'y':
                    if i + 1 < len(s) and _is_float(s[i + 1]):
                        create_trsf_option += ' -yrotation ' + s[i + 1]
                    else:
                        monitoring.to_log_and_console(proc + ": weird value for 'Y rotation' -> " + str(s[i + 1]), 1)
                elif s[i].lower() == 'z':
                    if i + 1 < len(s) and _is_float(s[i + 1]):
                        create_trsf_option += ' -zrotation ' + s[i + 1]
                    else:
                        monitoring.to_log_and_console(proc + ": weird value for 'Z rotation' -> " + str(s[i + 1]), 1)
                i += 2

            if len(create_trsf_option) > le:
                trsf_name = os.path.join(temporary_dir, "reference_transformation.trsf")
                cpp_wrapping.create_trsf(trsf_name, other_options=create_trsf_option, monitoring=monitoring)
                if os.path.isfile(trsf_name):
                    return trsf_name
                else:
                    monitoring.to_log_and_console(proc + ": unable to compute reference transformation", 1)
            else:
                monitoring.to_log_and_console(proc + ": unable to translate reference transformation angles", 1)

    return None


def _transformations_and_template(experiment, parameters, temporary_dir):
    """
    From transformations from one given image, compute the template to resample all images.
    :param experiment:
    :param parameters:
    :param temporary_dir:
    :return:
    """

    proc = "_transformations_and_template"

    if isinstance(parameters, IntraRegParameters) is False:
        monitoring.to_log_and_console(proc + ": bad type for 'parameters' parameter", 1)
        sys.exit(1)

    result_dir = _get_trsf_path(experiment)
    result_template = _get_template_path_name(experiment)

    if os.path.isfile(result_template) and monitoring.forceResultsToBeBuilt is not True \
            and parameters.rebuild_template is not True:
        return

    monitoring.to_log_and_console("       warning: this stage may be long", 2)

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    #
    # which format to use?
    # if requested, try to use segmentation image (threshold should be set to 2)
    # else use fusion images
    #
    template_format = None

    if parameters.template_type.lower() == 'segmentation' or parameters.template_type.lower() == 'seg':
        #
        # check whether segmentation image share a common suffix
        #
        path_template_format = os.path.join(experiment.get_embryo_path(), experiment.astec_dir.get_directory())
        suffix = common.get_file_suffix(experiment, path_template_format, experiment.astec_dir.get_image_format(),
                                        flag_time=experiment.get_time_format())

        if suffix is None:
            monitoring.to_log_and_console(proc + ": no consistent naming was found in '"
                                          + experiment.astec_dir.get_directory() + "'", 1)
            monitoring.to_log_and_console("\t switch to fused images as templates", 1)

        else:
            monitoring.to_log_and_console("       ... build template from segmentation images of '"
                                          + experiment.astec_dir.get_directory() + "'", 2)
            template_name = experiment.astec_dir.get_image_format() + '.' + suffix
            template_format = os.path.join(path_template_format, template_name)

    elif parameters.template_type.lower() == 'post-segmentation' \
            or parameters.template_type.lower() == 'post_segmentation' or parameters.template_type.lower() == 'post':
        #
        # check whether post-corrected segmentation image share a common suffix
        #
        path_template_format = os.path.join(experiment.get_embryo_path(), experiment.post_dir.get_directory())
        suffix = common.get_file_suffix(experiment, path_template_format, experiment.post_dir.get_image_format(),
                                        flag_time=experiment.get_time_format())

        if suffix is None:
            monitoring.to_log_and_console(proc + ": no consistent naming was found in '"
                                          + experiment.post_dir.get_directory() + "'", 1)
            monitoring.to_log_and_console("\t switch to fused images as templates", 1)

        else:
            monitoring.to_log_and_console("       ... build template from post-segmentation images of '"
                                          + experiment.post_dir.get_directory() + "'", 2)
            template_name = experiment.post_dir.get_image_format() + '.' + suffix
            template_format = os.path.join(path_template_format, template_name)

    #
    # use fusion images to build the template
    #
    if template_format is None:
        #
        # check whether fusion image share a common suffix
        #
        path_template_format = os.path.join(experiment.get_embryo_path(), experiment.fusion_dir.get_directory())
        suffix = common.get_file_suffix(experiment, path_template_format, experiment.fusion_dir.get_image_format(),
                                        flag_time=experiment.get_time_format())
        if suffix is None:
            monitoring.to_log_and_console(proc + ": no consistent naming was found in '"
                                          + experiment.fusion_dir.get_directory() + "'", 1)
            monitoring.to_log_and_console("\t Exiting", 1)
            sys.exit(1)
        else:
            monitoring.to_log_and_console("       ... build template from fusion images of '"
                                          + experiment.fusion_dir.get_directory() + "'", 2)
            template_name = experiment.fusion_dir.get_image_format() + '.' + suffix
            template_format = os.path.join(path_template_format, template_name)

    #
    # other parameters
    #

    first_time_point = experiment.first_time_point + experiment.delay_time_point
    last_time_point = experiment.last_time_point + experiment.delay_time_point

    trsf_format = _get_trsf_format(experiment)
    format_input = os.path.join(temporary_dir, trsf_format)
    format_output = os.path.join(result_dir, trsf_format)

    if parameters.reference_index is None:
        reference_index = first_time_point
    else:
        reference_index = parameters.reference_index

    #
    #
    #

    cpp_wrapping.change_multiple_trsfs(format_input, format_output, first_time_point, last_time_point, reference_index,
                                       result_template, trsf_type=parameters.registration.transformation_type,
                                       resolution=parameters.resolution, threshold=parameters.template_threshold,
                                       margin=parameters.margin, format_template=template_format,
                                       reference_transformation=_get_reference_transformation(parameters, temporary_dir),
                                       monitoring=monitoring)

    return


def _resample_images(experiment, parameters, template_image, directory_type, interpolation_mode='linear'):
    """
    resample all images given a set of transformations and a template
    :param experiment:
    :param parameters:
    :param template_image:
    :param directory_type:
    :param interpolation_mode:
    :return:
    """

    proc = "_resample_images"

    #
    # in case the template has been gziped, or copied into an other format
    #

    b = os.path.basename(template_image)
    d = os.path.dirname(template_image)
    local_template_name = common.find_file(d, b, file_type='image', callfrom=proc, local_monitoring=monitoring)
    if local_template_name is None:
        monitoring.to_log_and_console(proc + ": template '" + str(b) + "' was not found in '" + str(d) + "'", 1)
        monitoring.to_log_and_console("\t resampling will not be done")
        return
    local_template_image = os.path.join(d, local_template_name)

    #
    #
    #

    first_time_point = experiment.first_time_point + experiment.delay_time_point
    last_time_point = experiment.last_time_point + experiment.delay_time_point

    #
    #
    #

    if directory_type.lower() == 'fuse':
        working_dir = experiment.fusion_dir
    elif directory_type.lower() == 'post':
        working_dir = experiment.post_dir
    elif directory_type.lower() == 'seg':
        working_dir = experiment.astec_dir
    else:
        monitoring.to_log_and_console(proc + ": unknown directory type '" + str(directory_type) + "'", 1)
        monitoring.to_log_and_console("\t resampling will not be done")
        return

    #
    # loop on directories
    #

    trsf_dir = _get_trsf_path(experiment)

    for idir in range(working_dir.get_number_directories()):

        dir_input = working_dir.get_directory(idir)
        monitoring.to_log_and_console("     . resampling '" + str(dir_input) + "'", 2)
        dir_input = os.path.join(experiment.get_embryo_path(), working_dir.get_directory(idir))
        dir_output = os.path.join(experiment.intrareg_dir.get_directory(), working_dir.get_sub_directory(idir))
        if not os.path.isdir(dir_output):
            os.makedirs(dir_output)

        #
        # loop on images
        #

        for t in range(first_time_point + experiment.delay_time_point,
                       last_time_point + experiment.delay_time_point + 1, experiment.delta_time_point):

            input_name = working_dir.get_image_name(t)

            output_name = experiment.intrareg_dir.get_file_prefix() + experiment.intrareg_dir.get_file_suffix() + \
                          working_dir.get_file_suffix() + experiment.intrareg_dir.get_time_prefix() + \
                          experiment.get_time_index(t)

            output_image = common.find_file(dir_output, output_name, file_type='image', callfrom=proc,
                                            local_monitoring=None, verbose=False)

            if output_image is None or monitoring.forceResultsToBeBuilt is True or parameters.rebuild_template is True:

                input_image = common.find_file(dir_input, input_name, file_type='image', callfrom=proc,
                                               local_monitoring=monitoring)
                output_image = os.path.join(dir_output, output_name + '.' + str(experiment.result_image_suffix))
                trsf_name = os.path.join(trsf_dir, _get_trsf_name(experiment, t))

                if input_image is None:
                    monitoring.to_log_and_console(proc + ": image '" + str(input_name) + "' was not found in '"
                                                  + str(dir_input) + "'", 1)
                elif not os.path.isfile(trsf_name):
                    monitoring.to_log_and_console(proc + ": transformation '" + str(_get_trsf_name(experiment, t))
                                                  + "' was not found in '" + str(trsf_dir) + "'", 1)
                else:
                    monitoring.to_log_and_console("       resampling '" + str(input_image) + "'", 2)
                    input_image = os.path.join(dir_input, input_image)
                    cpp_wrapping.apply_transformation(input_image, output_image, trsf_name, local_template_image,
                                                      interpolation_mode=interpolation_mode,
                                                      cell_based_sigma=parameters.sigma_segmentation_images,
                                                      monitoring=monitoring)

    return


def _make_movies(experiment, parameters, directory_type, xylist, xzlist, yzlist):
    """

    :param experiment:
    :param parameters:
    :param directory_type:
    :param xylist:
    :param xzlist:
    :param yzlist:
    :return:
    """

    proc = "_make_movies"

    #
    #
    #

    first_time_point = experiment.first_time_point + experiment.delay_time_point
    last_time_point = experiment.last_time_point + experiment.delay_time_point

    #
    #
    #

    if directory_type.lower() == 'fuse':
        working_dir = experiment.fusion_dir
    elif directory_type.lower() == 'post':
        working_dir = experiment.post_dir
    elif directory_type.lower() == 'seg':
        working_dir = experiment.astec_dir
    else:
        monitoring.to_log_and_console(proc + ": unknown directory type '" + str(directory_type) + "'", 1)
        monitoring.to_log_and_console("\t movies will not be done")
        return

    #
    # should we switch to default behavior?
    #

    xy = []
    xz = []
    yz = []

    if len(xylist) > 0 or len(xzlist) > 0 or len(yzlist) > 0:
        xy = xylist
        xz = xzlist
        yz = yzlist

    #
    # loop on directories
    #

    for idir in range(working_dir.get_number_directories()):

        dir_input = os.path.join(experiment.intrareg_dir.get_directory(), working_dir.get_sub_directory(idir))
        monitoring.to_log_and_console("     . movies from '" + str(dir_input) + "'", 2)
        dir_output = os.path.join(experiment.intrareg_dir.get_directory(), 'MOVIES',
                                  working_dir.get_sub_directory(idir))

        if not os.path.isdir(dir_output):
            os.makedirs(dir_output)

        #
        # default behavior
        #
        if len(xy) == 0 and len(xz) == 0 and len(yz) == 0:
            #
            # read the first image and set XY slice in the middle
            #
            first_prefix = experiment.intrareg_dir.get_file_prefix() + experiment.intrareg_dir.get_file_suffix() + \
                          working_dir.get_file_suffix() + experiment.intrareg_dir.get_time_prefix() + \
                          experiment.get_time_index(first_time_point)
            first_name = common.find_file(dir_input, first_prefix, file_type='image', callfrom=proc,
                                          local_monitoring=None, verbose=False)
            if first_name is None:
                monitoring.to_log_and_console(proc + ": no file '" + str(first_prefix) + "' in '" + str(dir_input) +
                                              "'", 1)
                monitoring.to_log_and_console("\t movies will not be done")
                return
            first_image = imread(os.path.join(dir_input, first_name))
            xy.append(int(first_image.shape[2] / 2))
            del first_image

        input_format = experiment.intrareg_dir.get_file_prefix() + experiment.intrareg_dir.get_file_suffix() + \
                       working_dir.get_file_suffix() + experiment.intrareg_dir.get_time_prefix() + \
                       experiment.get_time_format()
        input_format = os.path.join(dir_input, input_format + '.' + str(experiment.result_image_suffix))

        #
        # processing
        #

        name_prefix = experiment.intrareg_dir.get_file_prefix() + experiment.intrareg_dir.get_file_suffix() + \
                      working_dir.get_file_suffix() + experiment.intrareg_dir.get_time_prefix() + \
                      experiment.get_time_index(first_time_point) + '-' + \
                      experiment.get_time_index(last_time_point)
        if len(xy) > 0:
            for s in xy:
                name_output = name_prefix + '_xy' + '{:0{width}d}'.format(s, width=4)
                name_output = os.path.join(dir_output, name_output + '.' + str(experiment.result_image_suffix))
                if os.path.isfile(name_output) is False or monitoring.forceResultsToBeBuilt is True \
                        or parameters.rebuild_template is True:
                    monitoring.to_log_and_console("       process xy=" + str(s), 2)
                    cpp_wrapping.crop_sequence(input_format, name_output, first_time_point, last_time_point, 'xy', s,
                                               monitoring=monitoring)

        if len(xz) > 0:
            for s in xz:
                name_output = name_prefix + '_xz' + '{:0{width}d}'.format(s, width=4)
                name_output = os.path.join(dir_output, name_output + '.' + str(experiment.result_image_suffix))
                if os.path.isfile(name_output) is False or monitoring.forceResultsToBeBuilt is True \
                        or parameters.rebuild_template is True:
                    monitoring.to_log_and_console("       process xz=" + str(s), 2)
                    cpp_wrapping.crop_sequence(input_format, name_output, first_time_point, last_time_point, 'xz', s,
                                               monitoring=monitoring)

        if len(yz) > 0:
            for s in yz:
                name_output = name_prefix + '_yz' + '{:0{width}d}'.format(s, width=4)
                name_output = os.path.join(dir_output, name_output + '.' + str(experiment.result_image_suffix))
                if os.path.isfile(name_output) is False or monitoring.forceResultsToBeBuilt is True \
                        or parameters.rebuild_template is True:
                    monitoring.to_log_and_console("       process yz=" + str(s), 2)
                    cpp_wrapping.crop_sequence(input_format, name_output, first_time_point, last_time_point, 'yz', s,
                                               monitoring=monitoring)

    return


def _make_maximum(experiment, directory_type):
    """

    :param experiment:
    :param directory_type:
    :return:
    """

    proc = "_make_maximum"

    #
    #
    #

    first_time_point = experiment.first_time_point + experiment.delay_time_point
    last_time_point = experiment.last_time_point + experiment.delay_time_point

    #
    #
    #

    if directory_type.lower() == 'fuse':
        working_dir = experiment.fusion_dir
    elif directory_type.lower() == 'post':
        working_dir = experiment.post_dir
    elif directory_type.lower() == 'seg':
        working_dir = experiment.astec_dir
    else:
        monitoring.to_log_and_console(proc + ": unknown directory type '" + str(directory_type) + "'", 1)
        monitoring.to_log_and_console("\t maximum will not be done")
        return

    #
    # loop on directories
    #

    for idir in range(working_dir.get_number_directories()):

        dir_input = os.path.join(experiment.intrareg_dir.get_directory(), working_dir.get_sub_directory(idir))
        monitoring.to_log_and_console("     . maximum from '" + str(dir_input) + "'", 2)
        dir_output = os.path.join(experiment.intrareg_dir.get_directory(), 'MAXIMUM',
                                  working_dir.get_sub_directory(idir))
        if not os.path.isdir(dir_output):
            os.makedirs(dir_output)

        input_format = experiment.intrareg_dir.get_file_prefix() + experiment.intrareg_dir.get_file_suffix() + \
                       working_dir.get_file_suffix() + experiment.intrareg_dir.get_time_prefix() + \
                       experiment.get_time_format()
        input_format = os.path.join(dir_input, input_format + '.' + str(experiment.result_image_suffix))

        #
        # processing
        #

        name_prefix = experiment.intrareg_dir.get_file_prefix() + experiment.intrareg_dir.get_file_suffix() + \
                      working_dir.get_file_suffix() + experiment.intrareg_dir.get_time_prefix() + \
                      experiment.get_time_index(first_time_point) + '-' + \
                      experiment.get_time_index(last_time_point)
        name_output = name_prefix + '_maximum'
        name_output = os.path.join(dir_output, name_output + '.' + str(experiment.result_image_suffix))
        cpp_wrapping.mean_images(input_format, name_output, first_time_point, last_time_point, operation="maximum",
                                 monitoring=monitoring)
    return

########################################################################################
#
#
#
########################################################################################


def intraregistration_control(experiment, parameters):
    """

    :param experiment:
    :param parameters:
    :return:
    """

    proc = "intraregistration_control"

    if isinstance(experiment, common.Experiment) is False:
        monitoring.to_log_and_console(proc + ": bad type for 'experiment' parameter", 1)
        sys.exit(1)
    if isinstance(parameters, IntraRegParameters) is False:
        monitoring.to_log_and_console(proc + ": bad type for 'parameters' parameter", 1)
        sys.exit(1)

    #
    # start processing
    #
    start_time = time.time()

    #
    # if template does not exists,
    # 1. compute transformations between successive images
    #    INTRAREG/INTRAREG_<EXP_INTRAREG>/CO-TRSFS directory
    # 2. compose transformations wrt a reference (default = first time point)
    #    INTRAREG/INTRAREG_<EXP_INTRAREG>/CO-TRSFS/TEMP directory
    # 3. re-compute transformations (ie translations) and template that includes all transformed images
    #    INTRAREG/INTRAREG_<EXP_INTRAREG>/TRSFS_t<first>-<last> directory
    #

    result_dir = _get_trsf_path(experiment)
    result_template = _get_template_path_name(experiment)

    if not os.path.isdir(result_dir) or os.path.isfile(result_template) \
            or parameters.rebuild_template is True or monitoring.forceResultsToBeBuilt is True:

        if experiment.delta_time_point > 1:
            monitoring.to_log_and_console(proc + ": warning, delta_time=" + str(experiment.delta_time_point)
                                          + ", this step may be fragile", 1)
        #
        # co-registration of any 2 successive images
        # will fill the INTRAREG/INTRAREG_<EXP_INTRAREG>/CO-TRSFS directory
        #

        monitoring.to_log_and_console("    .. co-registrations", 2)
        _coregistration_control(experiment, parameters.registration)

        #
        # composition of transformations by propagation
        # results in a set of transformation from the reference image (given by the reference_index)
        # towards each image
        # will fill the INTRAREG/INTRAREG_<EXP_INTRAREG>/TEMP directory
        #

        monitoring.to_log_and_console("    .. transformation composition", 2)

        temporary_dir = os.path.join(_get_cotrsf_path(experiment), 'TEMP')
        _transformations_from_reference(experiment, parameters, temporary_dir)

        #
        # re-composition of transformations
        # template creation for further resampling operations
        # will fill the INTRAREG/INTRAREG_<EXP_INTRAREG>/TRSFS_t<first>-<last> directory
        #

        monitoring.to_log_and_console("    .. transformation recomposition and template generation", 2)

        _transformations_and_template(experiment, parameters, temporary_dir)

        if monitoring.keepTemporaryFiles is False:
            shutil.rmtree(temporary_dir)

    #
    # template image and resampling transformations have been computed
    # resample images if required
    #
    # NOTE: il y a un probleme de coherence, puisque la premiere image de segmentation peut etre issue
    # de mars et donc etre nommee differemment. Pour le reechantillonage, on pourrait utiliser
    # reconstruction.get_segmentation_image().
    #

    if parameters.resample_fusion_images is True or parameters.movie_fusion_images is True \
            or len(parameters.xy_movie_fusion_images) > 0 or len(parameters.xz_movie_fusion_images) > 0 \
            or len(parameters.yz_movie_fusion_images) > 0:
        monitoring.to_log_and_console("    .. resampling fusion images", 2)
        _resample_images(experiment, parameters, result_template, 'fuse')

    if parameters.resample_segmentation_images is True or parameters.movie_segmentation_images is True \
            or len(parameters.xy_movie_segmentation_images) > 0 or len(parameters.xz_movie_segmentation_images) > 0 \
            or len(parameters.yz_movie_segmentation_images) > 0:
        monitoring.to_log_and_console("    .. resampling segmentation images", 2)
        _resample_images(experiment, parameters, result_template, 'seg', interpolation_mode='nearest')

    if parameters.resample_post_segmentation_images is True or parameters.movie_post_segmentation_images is True \
            or len(parameters.xy_movie_post_segmentation_images) > 0 \
            or len(parameters.xz_movie_post_segmentation_images) > 0 \
            or len(parameters.yz_movie_post_segmentation_images) > 0:
        monitoring.to_log_and_console("    .. resampling post-segmentation images", 2)
        _resample_images(experiment, parameters, result_template, 'post', interpolation_mode='nearest')

    #
    # make 3D=2D+t images = movie of evolving slices with respect to time
    #

    if parameters.movie_fusion_images is True or len(parameters.xy_movie_fusion_images) > 0 \
            or len(parameters.xz_movie_fusion_images) > 0 or len(parameters.yz_movie_fusion_images) > 0:
        monitoring.to_log_and_console("    .. movies from fusion images", 2)
        _make_movies(experiment, parameters, 'fuse', parameters.xy_movie_fusion_images,
                     parameters.xz_movie_fusion_images, parameters.yz_movie_fusion_images)

    if parameters.movie_segmentation_images is True or len(parameters.xy_movie_segmentation_images) > 0 \
            or len(parameters.xz_movie_segmentation_images) > 0 or len(parameters.yz_movie_segmentation_images) > 0:
        monitoring.to_log_and_console("    .. movies from segmentation images", 2)
        _make_movies(experiment, parameters, 'seg', parameters.xy_movie_segmentation_images,
                     parameters.xz_movie_segmentation_images, parameters.yz_movie_segmentation_images)

    if parameters.movie_post_segmentation_images is True or len(parameters.xy_movie_post_segmentation_images) > 0 \
            or len(parameters.xz_movie_post_segmentation_images) > 0 \
            or len(parameters.yz_movie_post_segmentation_images) > 0:
        monitoring.to_log_and_console("    .. movies from post-segmentation images", 2)
        _make_movies(experiment, parameters, 'post', parameters.xy_movie_post_segmentation_images,
                     parameters.xz_movie_post_segmentation_images, parameters.yz_movie_post_segmentation_images)

    #
    # make maximum images
    #

    if parameters.maximum_fusion_images is True:
        monitoring.to_log_and_console("    .. maximum from fusion images", 2)
        monitoring.to_log_and_console("       warning: this stage may be long", 2)
        _make_maximum(experiment, 'fuse')

    if parameters.maximum_segmentation_images is True:
        monitoring.to_log_and_console("    .. maximum from fusion images", 2)
        monitoring.to_log_and_console("       warning: this stage may be long", 2)
        _make_maximum(experiment, 'seg')

    if parameters.maximum_post_segmentation_images is True:
        monitoring.to_log_and_console("    .. maximum from fusion images", 2)
        monitoring.to_log_and_console("       warning: this stage may be long", 2)
        _make_maximum(experiment, 'post')

    #
    # end processing
    #

    end_time = time.time()

    monitoring.to_log_and_console('    computation time = ' + str(end_time - start_time) + ' s', 1)
    monitoring.to_log_and_console('', 1)

    return
