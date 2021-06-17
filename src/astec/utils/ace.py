
import os
import sys
import imp
import operator
import multiprocessing

import ASTEC.common as common
from ASTEC.CommunFunctions.ImageHandling import imread
import ASTEC.CommunFunctions.cpp_wrapping as cpp_wrapping

#
#
#
#
#


monitoring = common.Monitoring()

_instrumented_ = False

########################################################################################
#
# classes
# - computation parameters
#
# ### Membrane reconstruction parameters
# sigma_membrane=0.9: parameter for membranes enhancement filter (before the membranes binarization step)
#     (default, in real coordinates, adapted to images like Patrick/Ralph/Aquila)
#
# hard_thresholding=False (default): this option enables the user to set a "hard threshold"
#     (avoiding the automatic computation of anisotropic thresholds) if set to True.
#     In that case, the hard_threshold parameter is applied on the whole enhanced image
# hard_threshold=1.0 (default): if 'hard_thresholding' is set to True, this is the threshold for the
#     binarization of enhanced membranes image (1.0 : adapted for the time-point t001 of Aquila for example)
#
# manual=False (default): if set to True, this parameter activates the "manual" mode,
#     ie the user fixes the manual parameters for the thresholds computation for membranes binarization
# manual_sigma=7 (default): manual parameter for the initialization of Rayleigh function sigma parameter
#     for the directional histograms fitting for the thresholds computation
# sensitivity=0.99 (default) : sensitivity parameter for axial thresholds computation for membranes binarization,
#     with respect to the following criterion
#     (true positive rate) : threshold = #(membrane class>=threshold)/#(membrane class)
#
# sigma_TV=3.6: parameter of membranes propagation by the tensor voting algorithm
#     (default, real coordinates, adapted to images with a spatial resolution of 0.3um^3)
# sigma_LF=0.9  : parameter for the gaussian blurring of the reconstructed image by the tensor voting method
#     (default, real coordinates)
# sample=0.2 (default) : multiplicative parameter between 0 and 1 that fixes the sampling of voting token
#     from the initial image (1.0 means no resampling) (has an influence on the processing time)
#
# ### Intermediary images keep-or-leave parameters
# keep_membrane=False : if set to True, keeps all the images from membrane enhancement step (extrema and angles images)
#     (is automatically set to True if path_membrane_prefix is provided)
#
########################################################################################


class AceParameters(common.PrefixedParameter):

    ############################################################
    #
    # initialisation
    #
    ############################################################

    def __init__(self, prefix=None):
        common.PrefixedParameter.__init__(self, prefix=prefix)

        if "doc" not in self.__dict__:
            self.doc = {}

        doc = "\n"
        doc += "G(l)ace parameters overview:\n"
        doc += "============================\n"
        doc += "G(l)ace is a membrane-enhancement procedure. It works as follows\n"
        doc += "- an extrema image, corresponding to thethe membrane centers, is\n"
        doc += "  computed, thanks to a hessian based filtering\n"
        doc += "- these extrema are thresholded to get a binary image\n"
        doc += "  Thresholds are automatically computed from histograms of the\n"
        doc += "  extrema image"
        doc += "  - Gace: the threshold is computed globally\n"
        doc += "  - Glace: thresholds are computed on a cell-based manner\n"
        doc += "    One threshold is computed per cell, and the binary sub-images\n"
        doc += "    are fused afterwards. The cells are the cells at t-1 deformed on\n"
        doc += "    t. Works only in the propagation segmentation stage.\n"
        doc += "- the binary image is extended through a tensor voting procedure\n"
        doc += "- this last image is smoothed\n"
        doc += "\n"
        self.doc['ace_overview'] = doc

        #
        # parameters for filtering and membrane extrema extraction
        #
        doc = "\t Sigma of the gaussian used to compute the derivatives and thus\n"
        doc += "\t the extrema image. In real units.\n"
        self.doc['sigma_membrane'] = doc
        self.sigma_membrane = 0.9

        #
        # parameters for binarization
        #
        doc = "\t Possible values are True or False.\n"
        doc += "\t Should not be set to True.\n"
        doc += "\t If True, used the threshold 'hard_thresholding' to\n"
        doc += "\t binarize the extrema image\n"
        self.doc['hard_thresholding'] = doc
        self.hard_thresholding = False

        doc = "\t Threshold value when 'hard_thresholding' is set to True\n"
        self.doc['hard_threshold'] = doc
        self.hard_threshold = 1.0

        doc = "\t Possible values are True or False.\n"
        doc += "\t Should not be set to True.\n"
        doc += "\t If True, allows to tune the parameter 'manual_sigma' to compute \n"
        doc += "\t threshold from the histogram.\n"
        self.doc['manual'] = doc
        self.manual = False

        doc = "\t Axial histograms fitting initialization parameter\n"
        doc += "\t for the computation of membrane image binarization\n"
        doc += "\t Values have to be chosen in [5, 25]\n"
        self.doc['manual_sigma'] = doc
        self.manual_sigma = 15

        doc = "\t  membrane binarization parameter\n"
        self.doc['sensitivity'] = doc
        self.sensitivity = 0.99

        #
        # parameters for tensor voting
        #
        doc = "\t Sigma that defines the voting scale for membrane\n"
        doc += "\t In real units\n"
        self.doc['sigma_TV'] = doc
        self.sigma_TV = 1.08

        doc = "\t Sigma to smooth the image after tensor voting.\n"
        doc += "\t In real units\n"
        self.doc['sigma_LF'] = doc
        self.sigma_LF = 0.9

        doc = "\t fraction of the points used for tensor voting.\n"
        doc += "\t 1.0: all the points are used.\n"
        doc += "\t the more the points, the longest the computation.\n"
        self.doc['sample'] = doc
        self.sample = 0.2

        doc = "\t Random seed to be used when drawing points for the\n"
        doc += "\t tensor voting. Allows to reproduce an experiment when\n"
        doc += "\t 'sample' is less than 1.0. The used random seed can be \n"
        doc += "\t found in the log file.\n"
        self.doc['sample_random_seed'] = doc
        self.sample_random_seed = None

        #
        # for cell-based enhancement
        # dilation (in real unit)
        #
        doc = "\t Dilation radius for the cell bounding boxes\n"
        doc += "\t Used to compute local histograms\n"
        self.doc['bounding_box_dilation'] = doc
        self.bounding_box_dilation = 3.6

        #
        #
        #
        doc = "\t Number of processors for parallelization\n"
        self.doc['processors'] = doc
        self.processors = 7

    ############################################################
    #
    # print / write
    #
    ############################################################

    def print_parameters(self):
        print("")
        print('#')
        print('# AceParameters')
        print('#')
        print("")
        
        common.PrefixedParameter.print_parameters(self)

        for line in self.doc['ace_overview'].splitlines():
            print('# ' + line)

        self.varprint('sigma_membrane', self.sigma_membrane, self.doc['sigma_membrane'])

        self.varprint('hard_thresholding', self.hard_thresholding, self.doc['hard_thresholding'])
        self.varprint('hard_threshold', self.hard_threshold, self.doc['hard_threshold'])

        self.varprint('manual', self.manual, self.doc['manual'])
        self.varprint('manual_sigma', self.manual_sigma, self.doc['manual_sigma'])
        self.varprint('sensitivity', self.sensitivity, self.doc['sensitivity'])

        self.varprint('sigma_TV', self.sigma_TV, self.doc['sigma_TV'])
        self.varprint('sigma_LF', self.sigma_LF, self.doc['sigma_LF'])

        self.varprint('sample', self.sample, self.doc['sample'])
        self.varprint('sample_random_seed', self.sample_random_seed, self.doc['sample_random_seed'])

        self.varprint('bounding_box_dilation', self.bounding_box_dilation, self.doc['bounding_box_dilation'])
        self.varprint('processors', self.processors, self.doc['processors'])
        print("")
        return

    def write_parameters_in_file(self, logfile):
        logfile.write("\n")
        logfile.write("# \n")
        logfile.write("# AceParameters\n")
        logfile.write("# \n")
        logfile.write("\n")

        common.PrefixedParameter.write_parameters_in_file(self, logfile)

        for line in self.doc['ace_overview'].splitlines():
            logfile.write('# ' + line + '\n')

        self.varwrite(logfile, 'sigma_membrane', self.sigma_membrane, self.doc['sigma_membrane'])

        self.varwrite(logfile, 'hard_thresholding', self.hard_thresholding, self.doc['hard_thresholding'])
        self.varwrite(logfile, 'hard_threshold', self.hard_threshold, self.doc['hard_threshold'])

        self.varwrite(logfile, 'manual', self.manual, self.doc['manual'])
        self.varwrite(logfile, 'manual_sigma', self.manual_sigma, self.doc['manual_sigma'])
        self.varwrite(logfile, 'sensitivity', self.sensitivity, self.doc['sensitivity'])

        self.varwrite(logfile, 'sigma_TV', self.sigma_TV, self.doc['sigma_TV'])
        self.varwrite(logfile, 'sigma_LF', self.sigma_LF, self.doc['sigma_LF'])
        self.varwrite(logfile, 'sample', self.sample, self.doc['sample'])
        self.varwrite(logfile, 'sample_random_seed', self.sample_random_seed, self.doc['sample_random_seed'])

        self.varwrite(logfile, 'bounding_box_dilation', self.bounding_box_dilation, self.doc['bounding_box_dilation'])
        self.varwrite(logfile, 'processors', self.processors, self.doc['processors'])
        logfile.write("\n")
        return

    def write_parameters(self, log_file_name):
        with open(log_file_name, 'a') as logfile:
            self.write_parameters_in_file(logfile)
        return

    ############################################################
    #
    # update
    #
    ############################################################

    def update_from_parameters(self, parameters):
        self.sigma_membrane = self.read_parameter(parameters, 'sigma_membrane', self.sigma_membrane)

        self.hard_thresholding = self.read_parameter(parameters, 'hard_thresholding', self.hard_thresholding)
        self.hard_threshold = self.read_parameter(parameters, 'hard_threshold', self.hard_threshold)

        self.manual = self.read_parameter(parameters, 'manual', self.manual)
        self.manual_sigma = self.read_parameter(parameters, 'manual_sigma', self.manual_sigma)
        self.sensitivity = self.read_parameter(parameters, 'sensitivity', self.sensitivity)

        self.sigma_TV = self.read_parameter(parameters, 'sigma_TV', self.sigma_TV)
        self.sigma_LF = self.read_parameter(parameters, 'sigma_LF', self.sigma_LF)

        self.sample = self.read_parameter(parameters, 'sample', self.sample)
        self.sample_random_seed = self.read_parameter(parameters, 'sample_random_seed', self.sample_random_seed)

        self.bounding_box_dilation = self.read_parameter(parameters, 'bounding_box_dilation',
                                                         self.bounding_box_dilation)

        self.processors = self.read_parameter(parameters, 'processors', self.processors)

    def update_from_parameter_file(self, parameter_file):
        if parameter_file is None:
            return
        if not os.path.isfile(parameter_file):
            print("Error: '" + parameter_file + "' is not a valid file. Exiting.")
            sys.exit(1)

        parameters = imp.load_source('*', parameter_file)
        self.update_from_parameters(parameters)

    ############################################################
    #
    #
    #
    ############################################################

    def is_equal(self, p):
        if self.sigma_membrane != p.sigma_membrane:
            return False

        if self.hard_thresholding != p.hard_thresholding:
            return False
        if self.hard_threshold != p.hard_threshold:
            return False

        if self.manual != p.manual:
            return False
        if self.manual_sigma != p.manual_sigma:
            return False
        if self.sensitivity != p.sensitivity:
            return False

        if self.sigma_TV != p.sigma_TV:
            return False
        if self.sigma_LF != p.sigma_LF:
            return False
        if self.sample != p.sample:
            return False
        if self.sample_random_seed != p.sample_random_seed:
            return False

        if self.bounding_box_dilation != p.bounding_box_dilation:
            return False

        return True

########################################################################################
#
#
#
########################################################################################
#
# sigma_membrane=0.9, manual=False, manual_sigma=7, sensitivity=0.99, hard_thresholding=False,
# hard_threshold=1.0,
# sigma_TV=3.6, sigma_LF=0.9, sample=0.2,
# keep_membrane=False, keep_hist=False, keep_all=False, verbose=False):
# binary_input = False, path_bin = None,
# path_membrane_prefix = None,
#


def global_membrane_enhancement(path_input, path_output, experiment, binary_input=False,
                                temporary_path=None, parameters=None):
    """
    GACE for Global Automated Cell Extractor
    :param path_input:
    :param path_output:
    :param experiment:
    :param binary_input:
    :param temporary_path: 
    :param parameters: 
    :return: 
    """

    proc = 'global_membrane_enhancement'

    #
    # parameter checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, AceParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    if not os.path.isfile(path_input):
        monitoring.to_log_and_console(proc + ": input image does not exist", 0)
        monitoring.to_log_and_console("\t input image was '" + str(path_input) + "'", 0)
        monitoring.to_log_and_console("Exiting.", 0)
        sys.exit(1)

    #
    # set names
    # path_input is the full input image name
    # - e is the extension, including the '.' (e.g. '.inr')
    # - prefix_name is the basename without extension

    e = common.get_image_extension(path_input)
    b = os.path.basename(path_input)
    prefix_name = b[0:len(b) - len(e)]

    # tmp_prefix_name = temporary_path + os.path.sep + prefix_name

    #
    # if the output is already binary, there is no membrane extraction
    #

    path_tv_input = path_input

    if binary_input is False:

        #
        # get a binary image of membranes
        #
        # 1. surface extrema extraction
        #    it will generate (in the 'temporary_path' directory)
        #    - 'tmp_prefix_name'.ext
        #      the directional extrema image
        #    - 'tmp_prefix_name'.theta
        #    - 'tmp_prefix_name'.phi
        #      the two angle images that give the direction of the vector orthogonal to the membrane
        #
        ext_image = common.find_file(temporary_path, prefix_name + ".ext", file_type='image', callfrom=proc,
                                     local_monitoring=None, verbose=False)
        phi_image = common.find_file(temporary_path, prefix_name + ".phi", file_type='image', callfrom=proc,
                                     local_monitoring=None, verbose=False)
        theta_image = common.find_file(temporary_path, prefix_name + ".theta", file_type='image', callfrom=proc,
                                       local_monitoring=None, verbose=False)
        if ext_image is None or phi_image is None or theta_image is None or monitoring.forceResultsToBeBuilt is True:
            monitoring.to_log_and_console("       membrane extraction of '"
                                          + str(path_input).split(os.path.sep)[-1] + "'", 2)
            #
            # Local enhancement of membranes from 'path_input' image and extraction of the directional maxima
            # (generates the tmp_prefix_name+'.[ext|theta|phi].inr') images
            #
            tmp_prefix_name = os.path.join(temporary_path, prefix_name)
            cpp_wrapping.membrane_extraction(path_input, tmp_prefix_name, scale=parameters.sigma_membrane,
                                             other_options="-extension " + str(experiment.default_image_suffix),
                                             monitoring=monitoring)

        #
        # Membranes binarization
        #
        # 2. threshold
        #    it will generate (in the 'temporary_path' directory)
        #    - 'tmp_prefix_name'.bin
        #      the thresholded directional extrema image
        #    - 'tmp_prefix_name'.hist.txt
        #      a text file describing the image histogram
        #
        ext_image = common.find_file(temporary_path, prefix_name + ".ext", file_type='image', callfrom=proc,
                                     local_monitoring=monitoring)
        if ext_image is None:
            monitoring.to_log_and_console("       image '" + str(prefix_name + ".ext") + "' not found in '"
                                          + str(temporary_path) + "'", 2)
            monitoring.to_log_and_console("\t Exiting.")
            sys.exit(1)
        ext_image = os.path.join(temporary_path, ext_image)
        hist_file = os.path.join(temporary_path, prefix_name + ".histogram.txt")

        bin_image = common.find_file(temporary_path, prefix_name + ".bin", file_type='image', callfrom=proc,
                                     local_monitoring=None,
                                     verbose=False)
        if bin_image is None or monitoring.forceResultsToBeBuilt is True:
            if bin_image is None:
                bin_image = os.path.join(temporary_path, prefix_name + ".bin." + experiment.default_image_suffix)
            if parameters.hard_thresholding is True:
                #
                # hard threshold
                #
                monitoring.to_log_and_console("       membrane hard thresholding of '" +
                                              str(ext_image).split(os.path.sep)[-1] + "'", 2)
                cpp_wrapping.seuillage(path_input=ext_image, path_output=bin_image,
                                       low_threshold=parameters.hard_threshold, monitoring=monitoring)
            else:
                #
                # Anisotropic threshold of membranes (the choice of the sensitivity parameter may be critical)
                # Pay attention, the input name is the prefix name (without '.ext.inr')
                #
                monitoring.to_log_and_console("       membrane anisotropic thresholding of '" +
                                              str(ext_image).split(os.path.sep)[-1] + "'", 2)
                cpp_wrapping.anisotropic_histogram(path_input_extrema=ext_image, path_output_histogram=hist_file,
                                                   path_output=bin_image, manual=parameters.manual,
                                                   manual_sigma=parameters.manual_sigma,
                                                   sensitivity=parameters.sensitivity, monitoring=monitoring)

                if monitoring.keepTemporaryFiles is False:
                    os.remove(hist_file)
        else:
            bin_image = os.path.join(temporary_path, bin_image)
        #
        #
        #
        path_tv_input = bin_image

    #
    # Tensor voting on the image of binarized membranes
    #
    # it will generate (in the 'temporary_path' directory)
    #  - 'tmp_prefix_name'.bin.imvp1
    #  - 'tmp_prefix_name'.bin.imvp2
    #  - 'tmp_prefix_name'.bin.imvp3
    #    the three eigenvalue images generated by the tensor voting
    #  - 'tmp_prefix_name'.bin.tv
    #    the tensor voting scalar response (enhancing the membranes)
    #  - 'tmp_prefix_name'.bin.lf
    #    the previous image after smoothing (if smoothing is required)

    monitoring.to_log_and_console("       tensor voting from '"
                                  + str(path_tv_input).split(os.path.sep)[-1] + "'", 2)

    e = common.get_image_extension(path_tv_input)
    b = os.path.basename(path_tv_input)
    bin_name = b[0:len(b) - len(e)]

    bin_prefix_name = os.path.join(temporary_path, bin_name)

    if not os.path.isfile(path_output) or monitoring.forceResultsToBeBuilt is True:
        cpp_wrapping.tensor_voting_membrane(path_tv_input, bin_prefix_name, path_output,
                                            scale_tensor_voting=parameters.sigma_TV,
                                            sample=parameters.sample, random_seed=parameters.sample_random_seed,
                                            sigma_smoothing=parameters.sigma_LF, monitoring=monitoring)

    #
    # remove images
    #
    if binary_input is False:
        for suffix in ['.ext', '.phi', '.theta', '.bin']:
            tmp_image = common.find_file(temporary_path, prefix_name + suffix, file_type='image', callfrom=proc,
                                         local_monitoring=None, verbose=False)
            if tmp_image is not None and monitoring.keepTemporaryFiles is False:
                os.remove(os.path.join(temporary_path, tmp_image))
        tmp_image = common.find_file(temporary_path, prefix_name + ".histogram.txt", file_type=None, callfrom=proc,
                                     local_monitoring=None, verbose=False)
        if tmp_image is not None and monitoring.keepTemporaryFiles is False:
            os.remove(os.path.join(temporary_path, tmp_image))

    for suffix in ['.imvp1', '.imvp2', '.imvp3', '.tv', '.lf']:
        tmp_image = common.find_file(temporary_path, bin_name + suffix, file_type='image', callfrom=proc,
                                     local_monitoring=None, verbose=False)
        if tmp_image is not None and monitoring.keepTemporaryFiles is False:
            os.remove(os.path.join(temporary_path, tmp_image))

    return


########################################################################################
#
#
#
########################################################################################


def cell_binarization(parameters_for_parallelism):
    """

    :param parameters_for_parallelism:
    :return:
    """

    label_of_interest, bbox, previous_deformed_segmentation, tmp_prefix_name, parameters, default_image_suffix = \
        parameters_for_parallelism
    label_width = 5

    #
    # the GACE output images are cropped accordingly to the cell (ie label) bounding box
    #

    cellid = str('{:0{width}d}'.format(label_of_interest, width=label_width))
    cell_prefix_name = tmp_prefix_name + "_cell" + cellid

    cell_mask = cell_prefix_name + ".mask" + "." + default_image_suffix

    #
    # defines the dilated cell mask
    #

    cpp_wrapping.crop_image(previous_deformed_segmentation, cell_mask, bbox, monitoring=monitoring)
    cpp_wrapping.seuillage(cell_mask, cell_mask, low_threshold=label_of_interest, high_threshold=label_of_interest,
                           monitoring=monitoring)

    immask = imread(cell_mask)
    voxelsize = immask.voxelsize
    del immask

    #
    # bounding_box_dilation was already used for the bounding box dilation,
    # thus it remains consistent
    # default value for bounding_box_dilation is 3.6 um, thus 12 voxels
    # for an image of resolution 0.3 um
    #

    rx = int(parameters.bounding_box_dilation / voxelsize[0] + 0.5)
    ry = int(parameters.bounding_box_dilation / voxelsize[1] + 0.5)
    rz = int(parameters.bounding_box_dilation / voxelsize[2] + 0.5)
    dilation_radius = max(rx, ry, rz)
    cpp_wrapping.mathematical_morphology(cell_mask, cell_mask, other_options=" -dil -R " + str(dilation_radius),
                                         monitoring=monitoring)

    #
    # threshold extrema
    #

    full_ext = tmp_prefix_name + ".ext" "." + default_image_suffix
    cell_ext = cell_prefix_name + ".ext" + "." + default_image_suffix
    cell_hist = cell_prefix_name + ".hist" + ".txt"
    cell_bin = cell_prefix_name + ".bin" + "." + default_image_suffix

    cpp_wrapping.crop_image(full_ext, cell_ext, bbox, monitoring=monitoring)

    if parameters.hard_thresholding is True:
        #
        # hard threshold
        #
        cpp_wrapping.seuillage(path_input=cell_ext, path_output=cell_bin,
                               low_threshold=parameters.hard_threshold, monitoring=monitoring)
    else:
        full_theta = tmp_prefix_name + ".theta" + "." + default_image_suffix
        full_phi = tmp_prefix_name + ".phi" + "." + default_image_suffix
        cell_theta = cell_prefix_name + ".theta" + "." + default_image_suffix
        cell_phi = cell_prefix_name + ".phi" + "." + default_image_suffix

        cpp_wrapping.crop_image(full_theta, cell_theta, bbox, monitoring=monitoring)
        cpp_wrapping.crop_image(full_phi, cell_phi, bbox, monitoring=monitoring)

        cpp_wrapping.anisotropic_histogram(path_input_extrema=cell_ext, path_output_histogram=cell_hist,
                                           path_output=cell_bin, path_input_mask=cell_mask, manual=parameters.manual,
                                           manual_sigma=parameters.manual_sigma, sensitivity=parameters.sensitivity,
                                           monitoring=monitoring)
        if not _instrumented_:
            os.remove(cell_theta)
            os.remove(cell_phi)
            os.remove(cell_hist)

    if not _instrumented_:
        os.remove(cell_ext)

    cpp_wrapping.logical_operation(cell_mask, cell_bin, cell_bin, other_options="-mask", monitoring=monitoring)

    if not _instrumented_:
        os.remove(cell_mask)

    return label_of_interest, cell_bin


#
#
#


def cell_membrane_enhancement(path_input, previous_deformed_segmentation, path_output, experiment,
                              temporary_path=None, parameters=None):
    """

    :param path_input:
    :param previous_deformed_segmentation:
    :param path_output:
    :param experiment:
    :param temporary_path:
    :param parameters:
    :return:
    """

    proc = "cell_membrane_enhancement"

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, AceParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    # set names
    #

    e = common.get_image_extension(path_input)
    b = os.path.basename(path_input)
    prefix_name = b[0:len(b) - len(e)]

    tmp_prefix_name = os.path.join(temporary_path, prefix_name)

    #
    # first step
    # membrane extrema computation
    #
    #    it will generate (in the 'temporary_path' directory)
    #    - 'tmp_prefix_name'.ext
    #      the directional extrema image
    #    - 'tmp_prefix_name'.theta
    #    - 'tmp_prefix_name'.phi
    #      the two angle images that give the direction of the vector orthogonal to the membrane

    ext_image = common.find_file(temporary_path, prefix_name + '.ext', file_type='image', callfrom=proc,
                                 local_monitoring=None, verbose=False)
    phi_image = common.find_file(temporary_path, prefix_name + '.phi', file_type='image', callfrom=proc,
                                 local_monitoring=None, verbose=False)
    theta_image = common.find_file(temporary_path, prefix_name + '.theta', file_type='image', callfrom=proc,
                                   local_monitoring=None, verbose=False)

    if (ext_image is None or phi_image is None or theta_image is None or monitoring.forceResultsToBeBuilt is True):
        monitoring.to_log_and_console("       membrane extraction of '"
                                      + str(path_input).split(os.path.sep)[-1] + "'", 2)
        #
        # Local enhancement of membranes from 'path_input' image and extraction of the directional maxima
        # (generates the tmp_prefix_name+'.[ext|theta|phi].inr') images
        #
        cpp_wrapping.membrane_extraction(path_input, tmp_prefix_name,
                                         scale=parameters.sigma_membrane, monitoring=monitoring)

    #
    # second step
    # cell-based thresholding
    #

    #
    # bounding boxes of all labels
    # bboxes is a list of tuples [volume, xmin, ymin, zmin, xmax, ymax, zmax]
    #
    path_bboxes = common.add_suffix(previous_deformed_segmentation, '_bounding_boxes',
                                    new_dirname=temporary_path, new_extension='txt')
    bboxes = cpp_wrapping.bounding_boxes(previous_deformed_segmentation, path_bboxes=path_bboxes)

    #
    # dilation of bounding boxes
    # default value for bounding_box_dilation is 3.6 um
    # which yield 12 voxels for a image of resolution 0.3 um
    #
    if parameters.bounding_box_dilation > 0:

        imseg = imread(previous_deformed_segmentation)
        voxelsize = imseg.voxelsize
        xdim = imseg.shape[0]
        ydim = imseg.shape[1]
        zdim = imseg.shape[2]
        del imseg

        dx = int(parameters.bounding_box_dilation/voxelsize[0] + 0.5)
        if len(voxelsize) >= 2:
            dy = int(parameters.bounding_box_dilation / voxelsize[1] + 0.5)
        else:
            dy = int(parameters.bounding_box_dilation / voxelsize[0] + 0.5)
        if len(voxelsize) >= 3:
            dz = int(parameters.bounding_box_dilation / voxelsize[2] + 0.5)
        else:
            dz = int(parameters.bounding_box_dilation / voxelsize[0] + 0.5)

        dilation_operator = (0, -dx, -dy, -dz, dx, dy, dz)

        for x, b in bboxes.items():
            b = list(map(operator.add, b, dilation_operator))
            #
            # Note of Gael:
            # the coordinates of the "final" point of the bounding box can "go" out of the original image dimensions
            # without affecting the following of the program
            #
            if b[1] < 0:
                b[1] = 0
            if b[2] < 0:
                b[2] = 0
            if b[3] < 0:
                b[3] = 0
            if b[4] >= xdim:
                b[4] = xdim-1
            if b[5] >= ydim:
                b[5] = ydim-1
            if b[6] >= zdim:
                b[6] = zdim-1
            bboxes[x] = tuple(b)

    #
    # selection of cells to be enhanced
    # Gael choice was to define two parameters:
    # - labels_of_interest that can either be a list of labels, ie [3, 4, 7] or the string 'all'
    # - background that is nothing but [0,1], ie the two possible labels for the backgroung
    #
    # if type(labels_of_interest)==int:
    #     labels_of_interest=[labels_of_interest]
    # if type(labels_of_interest)==str and labels_of_interest=='all':
    #     labels_of_interest=[x for x in bboxes.keys() if not background.count(x)]
    #

    labels_of_interest = [x for x in bboxes if x != 0 and x != 1]

    #
    # parallel cell-based binarization
    # light_LACE METHOD INTERFACE (since ASTEC-170327) for each cell
    #

    monitoring.to_log_and_console(" ", 3)
    monitoring.to_log_and_console("       ----- start: cell-based parallel binarization -----", 3)

    pool = multiprocessing.Pool(processes=parameters.processors)
    mapping = []

    for label_of_interest in labels_of_interest:
        monitoring.to_log_and_console("       membrane binarization of cell '" + str(label_of_interest) + "'", 3)

        parameters_for_parallelism = (label_of_interest, bboxes[label_of_interest],
                                      previous_deformed_segmentation, tmp_prefix_name, parameters,
                                      experiment.default_image_suffix)
        # print str(parameters_for_parallelism)
        mapping.append(parameters_for_parallelism)

    outputs = pool.map(cell_binarization, mapping)
    pool.close()
    pool.terminate()

    monitoring.to_log_and_console("       ----- end:   cell-based parallel binarization -----", 3)
    monitoring.to_log_and_console(" ", 3)

    #
    # gather all cell binary images
    #

    bin_image = tmp_prefix_name + ".bin." + experiment.default_image_suffix
    cpp_wrapping.create_image(bin_image, tmp_prefix_name + ".ext." + experiment.default_image_suffix, "-o 1",
                              monitoring=monitoring)

    for binary_output_cell in outputs:
        cell_label = binary_output_cell[0]
        cell_bin = binary_output_cell[1]
        bbox = bboxes[cell_label]
        cpp_wrapping.patch_logical_operation(cell_bin, bin_image, bin_image, bbox, "-or", monitoring=monitoring)
        if not _instrumented_:
            os.remove(cell_bin)

    #
    # tensor voting
    #

    global_membrane_enhancement(bin_image, path_output, experiment, binary_input=True,
                                temporary_path=temporary_path, parameters=parameters)

    #
    # remove images
    #
    for suffix in ['.ext', '.phi', '.theta', '.bin']:
        tmp_image = common.find_file(temporary_path, prefix_name + suffix, file_type='image', callfrom=proc,
                                     local_monitoring=None, verbose=False)
        if tmp_image is not None and monitoring.keepTemporaryFiles is False:
            os.remove(os.path.join(temporary_path, tmp_image))

    return


########################################################################################
#
#
#
########################################################################################


def random_number():
    """
    Function which generates a random number made of 8 characters.
    :return:
    """
    import uuid
    return str(uuid.uuid4().fields[-1])[:5]+str(uuid.uuid4().fields[-1])[:5]


########################################################################################
#
#
#
########################################################################################


def light_LACE(parameters):
    """
    LACE method with
     - already transformed path_seg_0 (no need for path_fused_0 neither path_vector)
     - already computed image of membranes (no need for path_fused_1 )
    Interface :
        (path_mask, label_of_interest, path_membrane_prefix, path_bin, rayon_dil, manual, manual_sigma, hard_thresholding, hard_threshold, sensitivity, verbose) = parameters
     - path_mask : path to labelled image that defines the regions of interest (already transformed)
     - label_of_interest : label of interest of path_mask image. The region of interest is then dilated with a ray of value rayon_dil.
     - bbox : bounding box of the label of interest (before dilation) at the format [xmin, ymin, zmin, xmax, ymax, zmax].
     - path_membrane_prefix : paths of maxima of enhanced membranes image (ext) and of associated angles (theta, phi) --> path_membrane_prefix+".[ext|theta|phi].inr" must be existing.
     - path_bin : path to the binarised membranes result image
     - rayon_dil : dilation ray for region of interest (see path_mask and label_of_interest)
     - manual : if set to True, this parameter activates the "manual" mode, ie the user fixes the manual parameters for the thresholds computation for membranes binarization
                # parametre activant le mode manuel (ie parametrise) du seuil de binarisation des membranes si egal a True
     - manual_sigma : manual parameter for the initialization of Rayleigh function sigma parameter for the directional histograms fitting for the thresholds computation
                      # parametre manuel d'initialisation pour le calcul du seuil de binarisation des membranes
     - hard_thresholding : if set to True, this enables to fix a global threshold on the image. This threshold is given by the parameter 'hard_threshold'
                           # possibilite de choisir un seuillage dur global sur l'image en mettant cette option a True
     - hard_threshold : if 'hard_thresholding' is set to True, this is the threshold for the binarization of enhanced membranes image
                        # si hard_thresholding est a True, seuillage des membranes rehaussees via ce seuil
     - sensitivity : parameter which fixes the sensitivity criterion for axial thresholds computation:
                                (true positive rate) : threshold = #(membrane class>=threshold)/#(membrane class)
                     # parametre de calcul des seuils anisotropiques selon un critere de sensibilite
                     #          (true positive rate) : seuil = #(classe membrane>=seuil)/#(classe membrane)
     - verbose : verbosity of the function
    light_LACE return value is path_bin.
    """
    path_mask, label_of_interest, bbox, path_membrane_prefix, path_bin, rayon_dil, manual, manual_sigma, hard_thresholding, hard_threshold, sensitivity, verbose=parameters

    proc = "light_LACE"

    label_width = 5
    tmp_ID = str('{:0{width}d}'.format(label_of_interest, width=label_width))
    # tmp_ID = random_number()
    if _instrumented_:
        print(str(proc) + ": launch " + str(tmp_ID))
    path_WORK = os.path.dirname(path_mask).rstrip(os.path.sep)+os.path.sep
    path_mask_dil = path_WORK+'mask_at_1_dil_'+tmp_ID+'.inr'

    path_ext_tmp = path_WORK + "tmp_"+tmp_ID+'.ext.inr'
    path_theta_tmp = path_WORK + "tmp_"+tmp_ID+'.theta.inr'
    path_phi_tmp = path_WORK + "tmp_"+tmp_ID+'.phi.inr'

    if not path_bin:
        path_bin = path_WORK + "tmp_"+tmp_ID+"."+str(label_of_interest)+'.inr'

    if _instrumented_:
        print(" - " + str(tmp_ID) +": before assert")
        print("\t test" + str(path_membrane_prefix+".ext.inr"))
        print("\t test" + str(path_membrane_prefix + ".theta.inr"))
        print("\t test" + str(path_membrane_prefix + ".phi.inr"))


    assert ( os.path.exists(path_membrane_prefix+".ext.inr")) and ( os.path.exists(path_membrane_prefix+".theta.inr")) and ( os.path.exists(path_membrane_prefix+".phi.inr"))

    if _instrumented_:
        print(" - " + str(tmp_ID) +": before cell extraction")
    if bbox:
        cpp_wrapping.obsolete_cropImage(path_mask, path_mask_dil, bbox, verbose=verbose )
        cpp_wrapping.obsolete_seuillage(path_mask_dil, path_output=path_mask_dil,sb=label_of_interest, sh=label_of_interest, verbose=verbose )
    else:
        cpp_wrapping.obsolete_seuillage(path_mask, path_output=path_mask_dil,sb=label_of_interest, sh=label_of_interest, verbose=verbose )

    if _instrumented_:
        print(" - " + str(tmp_ID) +": before cell dilation")

    # Dilation of the ROI at t+1
    # Dilatation de la zone d'interet a l'instant t+1
    if True: # Computation of the dilation ray in real coordinates / Calcul du rayon de dilatation en coordonnees reelles
        rayon_dil /= imread(path_mask).voxelsize[0]
        rayon_dil = int(rayon_dil+0.5)
    cpp_wrapping.obsolete_morpho(path_mask_dil, path_mask_dil, ' -dil -R '+str(rayon_dil), verbose=verbose)

    # Here, path_mask_dil defines the ROI in which LACE should apply the segmentation
    # Ici, path_mask_dil definit la zone d'interet dans laquelle LACE doit realiser la segmentation

    if bbox:
        cpp_wrapping.obsolete_cropImage(path_membrane_prefix+".ext.inr", path_ext_tmp, bbox, verbose=verbose )

    if _instrumented_:
        print(" - " + str(tmp_ID) +": before binarization")

    # Membranes binarization
    if not hard_thresholding:
        # Anisotropic threshold of membranes (the choice of the sensitivity parameter may be critical)
        # Seuillage anisotropique des membranes (parametre de sensitivite potentiellement critique)
        if bbox:
            cpp_wrapping.obsolete_cropImage(path_membrane_prefix+".theta.inr", path_theta_tmp, bbox, verbose=verbose )
            cpp_wrapping.obsolete_cropImage(path_membrane_prefix+".phi.inr", path_phi_tmp, bbox, verbose=verbose )
            cpp_wrapping.obsolete_anisotropicHist(path_input=path_ext_tmp, path_output=path_bin, path_mask=path_mask_dil, manual=manual, manual_sigma=manual_sigma, sensitivity=sensitivity, keepAll=False, verbose=verbose)
        else:
            cpp_wrapping.obsolete_anisotropicHist(path_input=path_membrane_prefix+".ext.inr", path_output=path_bin, path_mask=path_mask_dil, manual=manual, manual_sigma=manual_sigma, sensitivity=sensitivity, keepAll=False, verbose=verbose)
    else:
        if bbox:
            cpp_wrapping.obsolete_seuillage(path_input=path_ext_tmp, path_output=path_bin,sb=hard_threshold, verbose=verbose)
        else:
            cpp_wrapping.obsolete_seuillage(path_input=path_membrane_prefix+".ext.inr", path_output=path_bin,sb=hard_threshold, verbose=verbose)

    # Mask application on the binary image
    cpp_wrapping.obsolete_Logic(path_mask_dil, path_bin, path_bin, Mode='mask', verbose=verbose)

    if not _instrumented_:
        if os.path.exists(path_mask_dil):
            cmd='rm ' + str(path_mask_dil)
            if verbose:
                print(cmd)
            os.system(cmd)
        if os.path.exists(path_ext_tmp):
            cmd='rm ' + str(path_ext_tmp)
            if verbose:
                print(cmd)
            os.system(cmd)
        if os.path.exists(path_theta_tmp):
            cmd='rm ' + str(path_theta_tmp)
            if verbose:
                print(cmd)
            os.system(cmd)
        if os.path.exists(path_phi_tmp):
            cmd='rm ' + str(path_phi_tmp)
            if verbose:
                print(cmd)
            os.system(cmd)

    return path_bin


def LACE(path_fused_0, path_fused_1, path_seg_0, label_of_interest, path_membrane_prefix=None, path_vector=None, path_bin=None, path_output=None, 
    rayon_dil=3.6, sigma_membrane=0.9, manual=False, manual_sigma=7, hard_thresholding=False, hard_threshold=1.0, sensitivity=0.99, sigma_TV=3.6, sigma_LF=0.9, sample=0.2,
    keep_membrane=False, keep_hist=False, keep_vector=False, keep_all=False, short_LACE=False, verbose=False):
    '''
    LACE for Local Automated Cell Extractor

    <ENGLISH>
    # Input paths:
    path_fused_0 : fused image at t (the one for which the segmentation is known)
    path_fused_1 : fused image at t+1 (the one that should be locally reconstructed)
    path_seg_0 : segmented image at t (must have the same dimensions as path_fused_0)
    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths to maxima image (ext) of enhanced images and its associated angle images which give the membranes spatial orientation (theta, phi), in order to avoid their recomputation if possible
    path_vector (optional) : path to the deformation field previously computed (via blockmatching for example) between t (flo) and t+1 (ref) : T_flo<-ref
                             If the path already exists, the deformation field is not recomputed, otherwise it is computed and kept

    # Label of interest at time t (or 0), which defines the propagated ROI at time t+1 (or 1)
    label_of_interest : integer that corresponds to the label of interest

    # Output paths:
    path_membrane_prefix+'.[ext|theta|phi].inr' (optional) : paths to save the maxima image (ext) of enhanced images and its associated angle images which give the membranes spatial orientation (theta, phi)
    path_vector (optional) : see description in the "Input paths" section
    path_bin (optional) : path to save the binarized membranes image (ie the image sent in input for the tensor voting step)
    path_output (optional) : path to save the output reconstructed image (default is None)

    # Mask parameters
    rayon_dil=3.6 (default, in real coordinates) : dilatation ray for propagated ROI from time t to t+1


    # Membrane reconstruction parameters
    sigma_membrane=0.9 (default, in real coordinates, adapted to images like Patrick/Ralph/Aquila) : parameter for membranes enhancement filter (before the membranes binarization step)
    sensitivity=0.99 (default) : sensitivity parameter for axial thresholds computation for membranes binarization, with respect to the following criterion
                                 (true positive rate) : threshold = #(membrane class>=threshold)/#(membrane class) 

    manual=False (default) : if set to True, this parameter activates the "manual" mode, ie the user fixes the manual parameters for the thresholds computation for membranes binarization
    manual_sigma=7 (default) : manual parameter for the initialization of Rayleigh function sigma parameter for the directional histograms fitting for the thresholds computation

    hard_thresholding=False (default) : This option enables the user to set a "hard threshold" (avoiding the automatic computation of anisotropic thresholds) if set to True.
                                        In that case, the hard_threshold parameter is applied on the whole enhanced image
    hard_threshold=1.0 (default) : if 'hard_thresholding' is set to True, this is the threshold for the binarization of enhanced membranes image
                              (1.0 : adapted for the time-point t001 of Aquila for example)

    sigma_TV=3.6 (default, real coordinates, adapted to images with a spatial resolution of 0.3um^3) : parameter of membranes propagation by the tensor voting algorithm
    sigma_LF=0.9 (default, real coordinates) : parameter for the gaussian blurring of the reconstructed image by the tensor voting method
    sample=0.2 (default) : multiplicative parameter between 0 and 1 that fixes the sampling of voting token from the initial image (1.0 means no resampling) (has an influence on the processing time)



    # Parameter for a shorter version of LACE (for its integration into GLACE framework)
    short_LACE (default=False) : if True, the tensor voting algorithm is not processed and the function returns the image of binarized images instead

    # Intermediary images keep-or-leave parameters
    keep_vector=False : if set to True, keeps the vectorfield transformation T_t<-t+1 which enables to resample the segmented image at t on the time-point t+1  (is automatically set to True if path_vector is provided)
    keep_membrane=False : if set to True, keeps all the images from membrane enhancement step (extrema and angles images) (is automatically set to True if path_membrane_prefix is provided)
    keep_hist=False : if set to True, keeps the file containing the axial histograms (file name ending with ".hist.txt")
    keep_all=False : if set to True, keeps all the intermediary images generated during the processing


    # Others
    verbose=False : verbosity of the function, displays for example the temporary files generation and deletion




    <FRENCH>
    # Paths d'entree
    path_fused_0 : image fusionnee a l'instant t (celle dont on connait la segmentation) ;
    path_fused_1 : image fusionnee a l'instant t+1 (celle qu'on souhaite reconstruire localement)
    path_seg_0 : image segmentee a l'instant t
    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths des images de maxima (ext) de membranes rehaussees et de leurs angles associes (theta, phi), afin d'eviter de les recalculer
    path_vector (optionel) : path vers le champ de deformation calcule (via blockmatching) entre t (flo) et t+1 (ref) : T_flo<-ref
                             Si le path existe deja, le champ de deformation n'est pas recalcule, sinon il est calcule et conserve

    # Label d'interet de l'instant 0, definissant la zone d'etude propagee a l'instant 1
    label_of_interest : integer

    # Path de sortie
    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths de sauvegarde des images de maxima (ext) de membranes rehaussees et de leurs angles associes (theta, phi)
    path_vector (optionel) : cf paths d'entree
    path_bin (optionel) : path de sauvegarde de l'image des membranes binarisees (image envoyee en entree de l'etape de vote de tenseurs)
    path_output (optionel) : path de sauvegarde de l'image reconstruite de sortie (par defaut : None)

    # Mask parameters
    rayon_dil=3.6 (defaut, exprime en reelles) : rayon de dilatation de la zone d'etude propagee a partir de l'instant t vers l'instant t+1

    # Membrane reconstruction parameters
    sigma_membrane=0.9 (defaut, unites reelles, adapte a des images de telles que Patrick/Ralph/Aquila) : parametre de rehaussement des
                                                                                   membranes avant binarisation de celles-ci
    sensitivity=0.99 (defaut) : parametre de calcul des seuils anisotropiques selon un critere de sensibilite
                                (true positive rate) : seuil = #(classe membrane>=seuil)/#(classe membrane) 
    manual=False (defaut) : parametre activant le mode manuel (ie parametrise) du seuil de binarisation des membranes si egal a True
    manual_sigma=7 (defaut) : parametre manuel d'initialisation pour le calcul du seuil de binarisation des membranes

    hard_thresholding=False (defaut) : Si echec de la precedente methode de seuillage des membranes par calcul automatique de seuils directionnels,
                              possibilite de choisir un seuillage dur global sur l'image en mettant cette option a True
    hard_threshold=1.0 (defaut) : Si hard_thresholding est a True, seuillage des membranes rehaussees via ce seuil
                              (1.0 : adaptee pour le time-point t001 d'Aquila par exemple)

    sigma_TV=3.6 (defaut, reelles, adapte a des images de resolution 0.3um^3) : parametre de propagation des membranes
                                                                                par votes de tenseurs
    sigma_LF=0.9 (defaut, reelles) : parametre de lissage gaussien de l'image des membranes reconstruite
    sample=0.2 (defaut) : echantillonne les votants de l'image initiale selon le coefficient (influe sur la vitesse de traitement)


    # version raccourcie de LACE (pour integration dans GLACE)
    short_LACE (default=False) : si True, ne procede pas au tensor voting et retourne l'image des membranes binarisees

    # Conserver ou non certaines images intermediaires
    keep_vector=False : conserver la transformation non lineaire T_t<-t+1 permettant de transformer l'image a t sur l'image a t+1 (se met a True automatiquement si path_vector est renseigne)
    keep_membrane=False : conserver toutes les images de l'etape de membranes (image d'extrema et d'angles) (se met a True automatiquement si path_membrane_prefix est renseigne)
    keep_hist=False : conserver le fichier d'histogrammes axiaux (nom de fichier se terminant par ".hist.txt")
    keep_all=False : conserver toutes les images intermediaires servant au calcul

    # Divers
    verbose=False : verbosite de la fonction, affiche par exemple les fichiers generes temporairement et ceux qui sont supprimes
    '''

    # Test for input images existence
    # Test existence des images d'entree
    assert(os.path.exists(path_fused_0) and os.path.exists(path_fused_1) and os.path.exists(path_seg_0))

    # Parameters for intermediary files keep-or-leave
    # Parametres de conservation des images generees
    if not keep_membrane:
        keep_membrane=keep_all
    if not keep_vector:
        keep_vector=keep_all
    if not keep_hist:
        keep_hist=keep_all

    # Definition of intermediary image paths
    # Definition des paths d'images intermediaires
    path_WORK=''
    if path_output:
        path_WORK = os.path.dirname(path_output).rstrip(os.path.sep)+os.path.sep
    else:
        if path_bin:
            path_WORK = os.path.dirname(path_bin).rstrip(os.path.sep)+os.path.sep
        else:
            path_WORK = os.path.dirname(path_fused_1).rstrip(os.path.sep)+os.path.sep
    if path_WORK==os.path.sep:
        path_WORK=os.path.curdir+os.path.sep

    tmp_ID = random_number()
    if path_vector:
        keep_vector = True
    else:
        path_vector = path_WORK+'tmp_vectorfield_'+tmp_ID+'.inr'
    if path_membrane_prefix:
        keep_membrane = True
    else:
        path_membrane_prefix=path_WORK+'tmp_membrane_'+tmp_ID+''

    path_mask = path_WORK+'mask_at_1_'+tmp_ID+'.inr.gz'
    path_mask_dil = path_WORK+'mask_at_1_dil_'+tmp_ID+'.inr.gz'
    path_affine_trsf = path_WORK+'tmp_affine_'+tmp_ID+'.txt'
    path_TV=path_WORK+'tmp_reconstructed_'+tmp_ID+'.inr'

    keep_tmp_bin=keep_all
    if not keep_tmp_bin:
        keep_tmp_bin=path_bin and (path_bin != path_membrane_prefix+'.bin.inr')

    # Verbose for files
    if verbose:
        print("Temporary files:")
        print(path_affine_trsf)
        if not keep_vector:
            print(path_vector)
        print(path_mask)
        if not keep_membrane:
            if keep_tmp_bin:
                print(path_membrane_prefix+".[ext|theta|phi].inr")
            else:
                print(path_membrane_prefix+".[bin|ext|theta|phi].inr")
        else:
            if not keep_tmp_bin:
                print(path_membrane_prefix+".bin.inr")
        if not keep_hist and not hard_thresholding:
            print(path_membrane_prefix+".hist.txt")
        print(path_TV)
    if verbose and (keep_vector or path_output):
        print("Output files:")
        if keep_vector:
            print(path_vector)
        if keep_membrane:
            if keep_tmp_bin:
                print(path_membrane_prefix+".[ext|theta|phi].inr")
            else:
                print(path_membrane_prefix+".[bin|ext|theta|phi].inr")
        else:
            if keep_tmp_bin:
                print(path_membrane_prefix+".bin.inr")
        if keep_hist and not hard_thresholding:
            print(path_membrane_prefix+".hist.txt")
        if path_bin and path_bin != path_membrane_prefix+".bin.inr":
            print(path_bin)
        if path_output:
            print(path_output)


    ### Output path ###
    if not os.path.isdir(path_WORK):
        try:
            os.mkdir(path_WORK)
        except Exception :
            print("Unexpected error: unable to create working directory")

    ### Stuff ###


    if not os.path.exists(path_vector):
        non_linear_registration(path_fused_0, path_fused_1, '/dev/null', path_affine_trsf, '/dev/null', path_vector, verbose=verbose)

    # ROI resampling at t+1
    apply_trsf(path_seg_0, path_trsf=path_vector, path_output=path_mask, template=path_fused_1, nearest=True, verbose=verbose)


    if (not os.path.exists(path_membrane_prefix+".ext.inr")) or (not os.path.exists(path_membrane_prefix+".theta.inr")) or (not os.path.exists(path_membrane_prefix+".phi.inr")):
        # Extraction of the ROI at t
        seuillage(path_mask, path_output=path_mask_dil,sb=label_of_interest, sh=label_of_interest, verbose=verbose )
        # Conversion in real coordinates of the dilation ray
        rayon_dil_vox = rayon_dil / imread(path_mask).voxelsize[0]
        rayon_dil_vox = int(rayon_dil_vox+0.5)
        morpho(path_mask_dil, path_mask_dil, ' -dil -R '+str(rayon_dil_vox), verbose=verbose)
        # Local enhancement of membranes from fused image at t+1 and extraction of the directional maxima (generates the '.[ext|theta|phi].inr')
        # Renforcement local des membranes de l'image fusionnee a l'instant t+1 + extraction des maxima directionnels
        cpp_wrapping.obsolete_membrane_renforcement(path_fused_1, prefix_output=path_membrane_prefix, path_mask=path_mask_dil,  init=sigma_membrane, verbose=verbose)
        if path_mask_dil and os.path.exists(path_mask_dil):
            cmd='rm ' + path_mask_dil
            if verbose:
                print(cmd)
            os.system(cmd)


    if not path_bin:
        path_bin=path_membrane_prefix+'.bin.inr'

    parameters=(path_mask, label_of_interest, None, path_membrane_prefix, path_bin, rayon_dil, manual, manual_sigma, hard_thresholding, hard_threshold, sensitivity, verbose)

    if verbose:
        print('Running light_LACE(' + str(parameters) + ') ...')

    path_bin=light_LACE(parameters)

    reconstructed_image_1=None
    if short_LACE:
        if path_bin:
            reconstructed_image_1=path_bin
        else:
            reconstructed_image_1=path_membrane_prefix+'.bin.inr'
    else:
        # Tensor voting on the binarized membranes image
        TVmembrane(path_input=path_bin, path_output=path_TV, sample=sample, scale=sigma_TV, sigma_LF=sigma_LF, realScale=True, keepAll=False, verbose=verbose)

        # Copie to the output path
        if path_output and path_TV != path_output:
            copy(path_TV, path_output, verbose=verbose)

        reconstructed_image_1=path_TV

    # Deletion of intermediary images
    files_to_rm = ""

    if not keep_all:
        if path_mask and os.path.exists(path_mask):
            files_to_rm += path_mask + " "
        if path_affine_trsf and os.path.exists(path_affine_trsf):
            files_to_rm += path_affine_trsf + " "
        if path_vector and os.path.exists(path_vector) and not keep_vector:
            files_to_rm += path_vector + " "
        if path_TV and path_TV != path_output and os.path.exists(path_TV):
            files_to_rm += path_TV + " "

    if not keep_membrane:
        if path_membrane_prefix:
            if os.path.exists(path_membrane_prefix+'.ext.inr'):
                files_to_rm += path_membrane_prefix+'.ext.inr' + " "
            if os.path.exists(path_membrane_prefix+'.theta.inr'):
                files_to_rm += path_membrane_prefix+'.theta.inr' + " "
            if os.path.exists(path_membrane_prefix+'.phi.inr'):
                files_to_rm += path_membrane_prefix+'.phi.inr' + " "

    if not keep_tmp_bin and path_membrane_prefix and os.path.exists(path_membrane_prefix+'.bin.inr'):
        files_to_rm +=  path_membrane_prefix+'.bin.inr' + " "

    if not keep_hist and path_membrane_prefix and os.path.exists(path_membrane_prefix+'.hist.txt'):
        files_to_rm += path_membrane_prefix+'.hist.txt' + " "

    if files_to_rm and not _instrumented_:
        if verbose:
            print("Deleting temporary files: \n" + files_to_rm)
        os.system("rm " + files_to_rm)

    # End of LACE
    return reconstructed_image_1





########################################################################################
#
#
#
########################################################################################

def GLACE(path_fused_0, path_fused_1, path_seg_0, labels_of_interest='all', background=[0,1], path_membrane_prefix=None, path_vector=None, path_bin=None, path_output=None, rayon_dil=3.6, 
    sigma_membrane=0.9, manual=False, manual_sigma=7, hard_thresholding=False, hard_threshold=1.0, sensitivity=0.99, sigma_TV=3.6, sigma_LF=0.9, sample=0.2,
    keep_membrane=False, keep_vector=False, keep_all=False,  nb_proc=7, verbose=False):
    '''
    GLACE --- Grouped Local Automated Cell Extractor

    Summary of the method steps :
        # Step 1 : non-linear registration between path_fused_0 and path_fused_1

        # Step 2 : membrane reinforcement on global image

        # Step 3 : init and build the binary image (loop of short LACEs)

            # Step 3.1 : LACE on the label

            # Step 3.2 : adding to the main binary image

        # Step 4 : GACE on the binary image

    GLACE for Grouped Local Automated Cell Extractor

    <ENGLISH>

    # Input paths:
    path_fused_0 : fused image at t (the one for which the segmentation is known)
    path_fused_1 : fused image at t+1 (the one that should be locally reconstructed)
    path_seg_0 : segmented image at t (must have the same dimensions as path_fused_0)
    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths to maxima image (ext) of enhanced images and its associated angle images which give the membranes spatial orientation (theta, phi), in order to avoid their recomputation if possible
    path_vector (optional) : path to the deformation field previously computed (via blockmatching for example) between t (flo) and t+1 (ref) : T_flo<-ref
                             If the path already exists, the deformation field is not recomputed, otherwise it is computed and kept

    # Labels of interest at time t (or 0), which define the propagated ROI at time t+1 (or 1)
    labels_of_interest : list of integers corresponding to labels of the segmented image at time t from which we want to reconstruct the membrane signal at time t+1. Possibly, one can provide a unique integer (equivalent to LACE).
                        Default : 'all' (means that we propagate all the labels which exist in path_seg_0). If 'all', the 'background' parameter fixes the background label(s) (default : [0,1]).


    # Output paths:
    path_membrane_prefix+'.[ext|theta|phi].inr' (optional) : paths to save the maxima image (ext) of enhanced images and its associated angle images which give the membranes spatial orientation (theta, phi)
    path_vector (optional) : see description in the "Input paths" section
    path_bin (optional) : path to save the binarized membranes image (ie the image sent in input for the tensor voting step)
    path_output (optional) : path to save the output reconstructed image (default is None)

    # Mask parameters
    rayon_dil=3.6 (default, in real coordinates) : dilatation ray for propagated ROI from time t to t+1


    # Membrane reconstruction parameters
    sigma_membrane=0.9 (default, in real coordinates, adapted to images like Patrick/Ralph/Aquila) : parameter for membranes enhancement filter (before the membranes binarization step)
    sensitivity=0.99 (default) : sensitivity parameter for axial thresholds computation for membranes binarization, with respect to the following criterion
                                 (true positive rate) : threshold = #(membrane class>=threshold)/#(membrane class) 

    manual=False (default) : if set to True, this parameter activates the "manual" mode, ie the user fixes the manual parameters for the thresholds computation for membranes binarization
    manual_sigma=7 (default) : manual parameter for the initialization of Rayleigh function sigma parameter for the directional histograms fitting for the thresholds computation

    hard_thresholding=False (default) : This option enables the user to set a "hard threshold" (avoiding the automatic computation of anisotropic thresholds) if set to True.
                                        In that case, the hard_threshold parameter is applied on the whole enhanced image
    hard_threshold=1.0 (default) : if 'hard_thresholding' is set to True, this is the threshold for the binarization of enhanced membranes image
                              (1.0 : adapted for the time-point t001 of Aquila for example)

    sigma_TV=3.6 (default, real coordinates, adapted to images with a spatial resolution of 0.3um^3) : parameter of membranes propagation by the tensor voting algorithm
    sigma_LF=0.9 (default, real coordinates) : parameter for the gaussian blurring of the reconstructed image by the tensor voting method
    sample=0.2 (default) : multiplicative parameter between 0 and 1 that fixes the sampling of voting token from the initial image (1.0 means no resampling) (has an influence on the processing time)



    # Parameter for a shorter version of LACE (for its integration into GLACE framework)
    short_LACE (default=False) : if True, the tensor voting algorithm is not processed and the function returns the image of binarized images instead

    # Intermediary images keep-or-leave parameters
    keep_vector=False : if set to True, keeps the vectorfield transformation T_t<-t+1 which enables to resample the segmented image at t on the time-point t+1  (is automatically set to True if path_vector is provided)
    keep_membrane=False : if set to True, keeps all the images from membrane enhancement step (extrema and angles images) (is automatically set to True if path_membrane_prefix is provided)
    keep_hist=False : if set to True, keeps the file containing the axial histograms (file name ending with ".hist.txt")
    keep_all=False : if set to True, keeps all the intermediary images generated during the processing

    # Parallelism
    nb_proc=7 : number of processes launched in parallel for LACE

    # Others
    verbose=False : verbosity of the function, displays for example the temporary files generation and deletion


    <FRENCH>

    # Paths d'entree
    path_fused_0 : image fusionnee a l'instant t (celle dont on connait la segmentation)
    path_fused_1 : image fusionnee a l'instant t+1 (celle qu'on souhaite reconstruire localement)
    path_seg_0 : image segmentee a l'instant t
    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths des images de maxima (ext) de membranes rehaussees et de leurs angles associes (theta, phi), afin d'eviter de les recalculer
    path_vector (optionel) : path vers le champ de deformation calcule (via blockmatching) entre t (flo) et t+1 (ref) : T_flo<-ref
                             Si le path existe deja, le champ de deformation n'est pas recalcule, sinon il est calcule et conserve

    # Label d'interet de l'instant 0, definissant la zone d'etude propagee a l'instant 1
    labels_of_interest : liste d'entiers correspondant aux labels de l'instant t dont on souhaite reconstruire le signal de membrane a t+1. Eventuellement, on peut ne donner qu'un entier (equivalent de LACE).
                        Defaut : 'all' (signifie qu'on propage tous les labels qui existent dans l'image path_seg_0). Si 'all', le parametre background fixe le ou les label(s) de fond (defaut : [0,1]).


    # Path de sortie
    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths de sauvegarde des images de maxima (ext) de membranes rehaussees et de leurs angles associes (theta, phi)
    path_vector (optionel) : cf paths d'entree
    path_bin (optionel) : path de sauvegarde de l'image des membranes binarisees (image envoyee en entree de l'etape de vote de tenseurs)
    path_output (optionel) : path de sauvegarde de l'image reconstruite de sortie (par defaut : None)

    # Mask parameters
    rayon_dil=3.6 (defaut, exprime en reelles) : rayon de dilatation de la zone d'etude propagee a partir de l'instant t vers l'instant t+1

    # Membrane reconstruction parameters
    sigma_membrane=0.9 (defaut, unites reelles, adapte a des images de telles que Patrick/Ralph/Aquila) : parametre de rehaussement des
                                                                                   membranes avant binarisation de celles-ci
    sensitivity=0.99 (defaut) : parametre de calcul des seuils anisotropiques selon un critere de sensibilite
                                (true positive rate) : seuil = #(classe membrane>=seuil)/#(classe membrane) 
    manual=False (defaut) : parametre activant le mode manuel (ie parametrise) du seuil de binarisation des membranes si egal a True
    manual_sigma=7 (defaut) : parametre manuel d'initialisation pour le calcul du seuil de binarisation des membranes

    hard_thresholding=False (defaut) : Si echec de la precedente methode de seuillage des membranes par calcul automatique de seuils directionnels,
                              possibilite de choisir un seuillage dur global sur l'image en mettant cette option a True
    hard_threshold=1.0 (defaut) : Si hard_thresholding est a True, seuillage des membranes rehaussees via ce seuil
                              (1.0 : adaptee pour le time-point t001 d'Aquila par exemple)

    sigma_TV=3.6 (defaut, reelles, adapte a des images de resolution 0.3um^3) : parametre de propagation des membranes
                                                                                par votes de tenseurs 
    sigma_LF=0.9 (defaut, reelles) : parametre de lissage gaussien de l'image des membranes reconstruite
    sample=0.2 (defaut) : echantillonne les votants de l'image initiale selon le coefficient (influe sur la vitesse de traitement)

    # Conserver ou non certaines images intermediaires
    keep_vector=False : conserver la transformation non lineaire T_t<-t+1 permettant de transformer l'image a t sur l'image a t+1 (se met a True automatiquement si path_vector est renseigne)
    keep_membrane=False : conserver toutes les images de l'etape de membranes (image d'extrema et d'angles) (se met a True automatiquement si path_membrane_prefix est renseigne)
    keep_all=False : conserver toutes les images intermediaires servant au calcul

    # Divers
    verbose=False : verbosite de la fonction, affiche par exemple les fichiers generes temporairement et ceux qui sont supprimes

    # Parallelisme
    nb_proc=7 : nombre de processus lances en parallele pour LACE

    '''

    # Step 1 : non-linear registration between path_fused_0 and path_fused_1

    # Step 2 : membrane reinforcement on global image

    # Step 3 : init and build the binary image (loop of short LACEs)

        # Step 3.1 : LACE on the label

        # Step 3.2 : adding to the main binary image

    # Step 4 : GACE on the binary image


    # Test for input images existence
    # Test existence des images d'entree
    assert os.path.exists(path_fused_0), 'Miss file '+path_fused_0
    assert os.path.exists(path_fused_1), 'Miss file '+path_fused_1
    assert os.path.exists(path_seg_0), 'Miss file '+path_seg_0

    # Parameters for intermediary files keep-or-leave
    # Parametres de conservation des images generees
    if not keep_membrane:
        keep_membrane=keep_all
    if not keep_vector:
        keep_vector=keep_all

    # Definition of intermediary image paths
    # Definition des paths d'images intermediaires
    path_WORK=''
    if path_output:
        path_WORK = os.path.dirname(path_output).rstrip(os.path.sep)+os.path.sep
    else:
        path_WORK = os.path.dirname(path_fused_0).rstrip(os.path.sep)+os.path.sep
    if path_WORK==os.path.sep:
        path_WORK=os.path.curdir+os.path.sep

    tmp_ID = random_number()

    keep_output=True
    if not path_output:
        keep_output=False
        path_output = path_WORK+'tmp_output_'+tmp_ID+'.inr'

    path_affine_trsf = path_WORK+'tmp_affine_'+tmp_ID+'.txt'

    if path_vector:
        keep_vector = True
    else:
        path_vector = path_WORK+'tmp_vectorfield_'+tmp_ID+'.inr'
    if path_membrane_prefix:
        keep_membrane = True
    else:
        path_membrane_prefix=path_WORK+'tmp_membrane_'+tmp_ID+''

    path_seg_trsf = path_WORK+'seg_trsf_at_1_'+tmp_ID+'.inr'

    ### Output path ###
    if not os.path.isdir(path_WORK):
        try:
            os.mkdir(path_WORK)
        except Exception :
            print("Unexpected error: unable to create working directory")

    rayon_dil_voxel=None

    if rayon_dil:	# dilation of all the bounding boxes
        rayon_dil_voxel = rayon_dil / imread(path_fused_1).voxelsize[0]
        rayon_dil_voxel = int(rayon_dil_voxel+0.5)

    ### Stuff ###

    # Step 1

    if not os.path.exists(path_vector):
        non_linear_registration(path_fused_0, path_fused_1, '/dev/null', path_affine_trsf, '/dev/null', path_vector, verbose=verbose)

    # Projection of ROI at t+1

    apply_trsf(path_seg_0, path_trsf=path_vector, path_output=path_seg_trsf, template=path_fused_1, nearest=True, verbose=verbose)

    # Deletion of temporary files
    files_to_rm = path_affine_trsf + ' '
    if not keep_vector:
        files_to_rm += path_vector + ' '
    if files_to_rm and not _instrumented_:
        if verbose:
            print("Deleting temporary files: \n" + files_to_rm)
        os.system("rm " + files_to_rm)

    # GLACE_from_resampled_segmentation for the steps 2 to 4:
    # - global membrane enhancement
    # - light_LACE call for each ROI
    # - Union of binarised image patches
    # - tensor voting process (GACE)
    print("\n##################################################################### \
           \n############# Calling GLACE_from_resampled_segmentation ############# \
           \n##################################################################### \n")

    GLACE_from_resampled_segmentation(path_fused_1, path_seg_trsf, labels_of_interest=labels_of_interest, background=background, path_membrane_prefix=path_membrane_prefix, path_bin=path_bin, path_output=path_output, rayon_dil=rayon_dil,
    sigma_membrane=sigma_membrane, manual=manual, manual_sigma=manual_sigma, hard_thresholding=hard_thresholding, hard_threshold=hard_threshold, sensitivity=sensitivity, sigma_TV=sigma_TV, sigma_LF=sigma_LF, sample=sample,
    keep_membrane=True, keep_all=keep_all,  nb_proc=nb_proc, verbose=verbose)

    print("\n################################################ \
           \n############# Back to GLACE method ############# \
           \n################################################ \n")

    reconstructed_image_1=imread(path_output)

    # Deletion of temporary files
    files_to_rm=""

    if os.path.exists(path_membrane_prefix+'.hist.txt'):
        files_to_rm += path_membrane_prefix+'.hist.txt '

    if os.path.exists(path_membrane_prefix+'.bin.inr'):
        files_to_rm += path_membrane_prefix+'.bin.inr '

    if os.path.exists(path_seg_trsf):
        files_to_rm += path_seg_trsf + ' '

    if not keep_membrane:
        if os.path.exists(path_membrane_prefix+'.ext.inr'):
            files_to_rm += path_membrane_prefix+'.ext.inr '
        if os.path.exists(path_membrane_prefix+'.theta.inr'):
            files_to_rm += path_membrane_prefix+'.theta.inr '
        if os.path.exists(path_membrane_prefix+'.phi.inr'):
            files_to_rm += path_membrane_prefix+'.phi.inr '

    if files_to_rm and not _instrumented_:
        if verbose:
            print("Deleting temporary files: \n" + files_to_rm)
        os.system("rm " + files_to_rm)

    return reconstructed_image_1

########################################################################################
#
#
#
########################################################################################

def GLACE_from_resampled_segmentation(path_fused_1, path_seg_trsf, labels_of_interest='all', background=[0,1], path_membrane_prefix=None, path_bin=None, path_output=None, rayon_dil=3.6,
    sigma_membrane=0.9, manual=False, manual_sigma=7, hard_thresholding=False, hard_threshold=1.0, sensitivity=0.99, sigma_TV=3.6, sigma_LF=0.9, sample=0.2,
    keep_membrane=False, keep_all=False,  nb_proc=7, verbose=False):
    '''
    GLACE_from_resampled_segmentation --- Grouped Local Automated Cell Extractor sub-process

    Summary of the method steps :
        # Step 1 : non-linear registration between path_fused_0 and path_fused_1

        # Step 2 : membrane reinforcement on global image

        # Step 3 : init and build the binary image (loop of short LACEs)

            # Step 3.1 : LACE on the label

            # Step 3.2 : adding to the main binary image

        # Step 4 : GACE on the binary image

    GLACE for Grouped Local Automated Cell Extractor

    # Input paths:
    path_fused_1 : fused image to be processed (the one that should be locally reconstructed)
    path_seg_trsf : segmented image defining the ROIs for GLACE method (must have the same dimensions as path_fused_1)
    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths to maxima image (ext) of enhanced images and its associated angle images which give the membranes spatial orientation (theta, phi), in order to avoid their recomputation if possible
    path_vector (optional) : path to the deformation field previously computed (via blockmatching for example) between t (flo) and t+1 (ref) : T_flo<-ref
                             If the path already exists, the deformation field is not recomputed, otherwise it is computed and kept

    # Labels of interest at time t (or 0), which define the propagated ROI at time t+1 (or 1)
    labels_of_interest : list of integers corresponding to labels of the segmented image at time t from which we want to reconstruct the membrane signal at time t+1. Possibly, one can provide a unique integer (equivalent to LACE).
                        Default : 'all' (means that we propagate all the labels which exist in path_seg_0). If 'all', the 'background' parameter fixes the background label(s) (default : [0,1]).


    # Output paths:
    path_membrane_prefix+'.[ext|theta|phi].inr' (optional) : paths to save the maxima image (ext) of enhanced images and its associated angle images which give the membranes spatial orientation (theta, phi)
    path_bin (optional) : path to save the binarized membranes image (ie the image sent in input for the tensor voting step)
    path_output (optional) : path to save the output reconstructed image (default is None)

    # Mask parameters
    rayon_dil=3.6 (default, in real coordinates) : dilatation ray for propagated ROI from time t to t+1


    # Membrane reconstruction parameters
    sigma_membrane=0.9 (default, in real coordinates, adapted to images like Patrick/Ralph/Aquila) : parameter for membranes enhancement filter (before the membranes binarization step)
    sensitivity=0.99 (default) : sensitivity parameter for axial thresholds computation for membranes binarization, with respect to the following criterion
                                 (true positive rate) : threshold = #(membrane class>=threshold)/#(membrane class) 

    manual=False (default) : if set to True, this parameter activates the "manual" mode, ie the user fixes the manual parameters for the thresholds computation for membranes binarization
    manual_sigma=7 (default) : manual parameter for the initialization of Rayleigh function sigma parameter for the directional histograms fitting for the thresholds computation

    hard_thresholding=False (default) : This option enables the user to set a "hard threshold" (avoiding the automatic computation of anisotropic thresholds) if set to True.
                                        In that case, the hard_threshold parameter is applied on the whole enhanced image
    hard_threshold=1.0 (default) : if 'hard_thresholding' is set to True, this is the threshold for the binarization of enhanced membranes image
                              (1.0 : adapted for the time-point t001 of Aquila for example)

    sigma_TV=3.6 (default, real coordinates, adapted to images with a spatial resolution of 0.3um^3) : parameter of membranes propagation by the tensor voting algorithm
    sigma_LF=0.9 (default, real coordinates) : parameter for the gaussian blurring of the reconstructed image by the tensor voting method
    sample=0.2 (default) : multiplicative parameter between 0 and 1 that fixes the sampling of voting token from the initial image (1.0 means no resampling) (has an influence on the processing time)



    # Parameter for a shorter version of LACE (for its integration into GLACE framework)
    short_LACE (default=False) : if True, the tensor voting algorithm is not processed and the function returns the image of binarized images instead

    # Intermediary images keep-or-leave parameters
    keep_membrane=False : if set to True, keeps all the images from membrane enhancement step (extrema and angles images) (is automatically set to True if path_membrane_prefix is provided)
    keep_hist=False : if set to True, keeps the file containing the axial histograms (file name ending with ".hist.txt")
    keep_all=False : if set to True, keeps all the intermediary images generated during the processing

    # Parallelism
    nb_proc=7 : number of processes launched in parallel for LACE

    # Others
    verbose=False : verbosity of the function, displays for example the temporary files generation and deletion

    #################################################################

    # Main steps of this process are:
    # * membrane reinforcement on global image

    # * init and build the binary image (loop of short LACEs)

    #	* LACE on the label

    #	* adding to the main binary image

    # * GACE on the binary image

    '''

    proc = "GLACE_from_resampled_segmentation"

    # Test for input images existence
    # Test existence des images d'entree
    assert os.path.exists(path_fused_1), 'Miss fusion file '+path_fused_1
    assert os.path.exists(path_seg_trsf), 'Miss segmented file '+path_seg_trsf

    # Multi process import
    from multiprocessing import Process, Queue, Pool

    if type(labels_of_interest)==int:
        labels_of_interest=[labels_of_interest]


    # Parameters for intermediary files keep-or-leave
    # Parametres de conservation des images generees
    if not keep_membrane:
        keep_membrane=keep_all

    # Definition of intermediary image paths
    # Definition des paths d'images intermediaires
    path_WORK=''
    if path_output:
        path_WORK = os.path.dirname(path_output).rstrip(os.path.sep)+os.path.sep
    else:
        path_WORK = os.path.dirname(path_fused_1).rstrip(os.path.sep)+os.path.sep
    if path_WORK==os.path.sep:
        path_WORK=os.path.curdir+os.path.sep

    tmp_ID = random_number()

    keep_output=True
    if not path_output:
        keep_output=False
        path_output = path_WORK+'tmp_output_'+tmp_ID+'.inr'

    path_affine_trsf = path_WORK+'tmp_affine_'+tmp_ID+'.txt'

    if path_membrane_prefix:
        keep_membrane = True
    else:
        path_membrane_prefix=path_WORK+'tmp_membrane_'+tmp_ID+''

    ### Output path ###
    if not os.path.isdir(path_WORK):
        try:
            os.mkdir(path_WORK)
        except Exception :
            print("Unexpected error: unable to create working directory")


    # First step

    if (not os.path.exists(path_membrane_prefix+".ext.inr")) or (not os.path.exists(path_membrane_prefix+".theta.inr")) or (not os.path.exists(path_membrane_prefix+".phi.inr")):
        # Local enhancement of membranes from fused image at t+1 and extraction of the directional maxima (generates the '.[ext|theta|phi].inr')
        # Renforcement des membranes de l'image fusionnee a l'instant t+1 + extraction des maxima directionnels
        if _instrumented_ :
            print(str(proc) + ": call obsolete_membrane_renforcement()")
        cpp_wrapping.obsolete_membrane_renforcement(path_fused_1, prefix_output=path_membrane_prefix, path_mask=None,  init=sigma_membrane, verbose=verbose)

    # Second step

    bboxes=cpp_wrapping.obsolete_boudingboxes(path_seg_trsf, verbose=verbose)
    if rayon_dil:	# dilation of all the bounding boxes
        rayon_dil_voxel = rayon_dil / imread(path_fused_1).voxelsize[0]
        rayon_dil_voxel = int(rayon_dil_voxel+0.5)
        dilation_tuple = (0, -rayon_dil_voxel, -rayon_dil_voxel, -rayon_dil_voxel, rayon_dil_voxel, rayon_dil_voxel, rayon_dil_voxel)
        import operator
        for x, b in bboxes.items():
            b=list(map(operator.add, b, dilation_tuple))
            if b[1] < 0:	# origin in x >= 0
                b[1] = 0
            if b[2] < 0:	# origin in y >= 0
                b[2] = 0
            if b[3] < 0:	# origin in z >= 0
                b[3] = 0
            # NB : the coordinates of the "final" point of the bounding box can "go" out of the original image dimensions without affecting the following of the program
            # NB : les coordonnees du point "final" de la bounding box peuvent aller au dela de la dimension de l'image d'origine sans que cela n'affecte la suite du programme
            bboxes[x]=tuple(b)

    if type(labels_of_interest)==str and labels_of_interest=='all':
        labels_of_interest=[x for x in bboxes if not background.count(x)]

    pool=Pool(processes=nb_proc)
    mapping=[]

    for label_of_interest in labels_of_interest:

        # Second step part 1
        # 	light_LACE METHOD INTERFACE (since ASTEC-170327):
        parameters=(path_seg_trsf, label_of_interest, bboxes[label_of_interest], path_membrane_prefix, None, \
            rayon_dil, manual, manual_sigma, hard_thresholding, hard_threshold, sensitivity, \
            verbose)
        if verbose:
            print('Running light_LACE(' + str(parameters) + ') ...')

        mapping.append(parameters)


    outputs=pool.map(light_LACE, mapping)
    pool.close()
    pool.terminate()

    path_union_of_local_bins=path_membrane_prefix+'.bin_union.inr'
    cpp_wrapping.obsolete_createImage(path_union_of_local_bins, path_fused_1, '-o 1', verbose=verbose)

    for path_local_bin in outputs:
        # Second step part 2
        label_of_interest = int(path_local_bin.split(os.path.sep)[-1].split('.')[-2])
        bbox=bboxes[label_of_interest]
        cpp_wrapping.obsolete_patchLogic(path_local_bin, path_union_of_local_bins, path_union_of_local_bins, bbox, Mode='or', verbose=verbose)
        if not keep_all:
            cmd='rm ' + path_local_bin
            print(cmd)
            os.system(cmd)

    keep_union_of_local_bins=False
    if path_bin:
        if path_bin != path_union_of_local_bins:
            keep_union_of_local_bins=False
            cpp_wrapping.obsolete_copy(path_union_of_local_bins, path_bin, verbose)
        else:
            keep_union_of_local_bins=True

    # Third step
    # GACE on the binary image (tensor voting)

    GACE(path_union_of_local_bins, binary_input=True, path_output=path_output,
        sigma_membrane=sigma_membrane, manual=manual, manual_sigma=manual_sigma, sensitivity=sensitivity, hard_thresholding=hard_thresholding, hard_threshold=hard_threshold, sigma_TV=sigma_TV, sigma_LF=sigma_LF, sample=sample,
         keep_membrane=True, keep_all=False, verbose=verbose)

    # Deletion of temporary files
    files_to_rm=""

    if os.path.exists(path_membrane_prefix+'.hist.txt'):
        files_to_rm += path_membrane_prefix+'.hist.txt '

    if os.path.exists(path_membrane_prefix+'.bin.inr'):
        files_to_rm += path_membrane_prefix+'.bin.inr '

    if not keep_membrane:
        files_to_rm += path_membrane_prefix+'.ext.inr '
        files_to_rm += path_membrane_prefix+'.theta.inr '
        files_to_rm += path_membrane_prefix+'.phi.inr '
    if not keep_union_of_local_bins:
        files_to_rm += path_membrane_prefix+'.bin_union.inr '

    if files_to_rm and not _instrumented_:
        if verbose:
            print("Deleting temporary files: \n" + files_to_rm)
        os.system("rm " + files_to_rm)

########################################################################################
#
#
#
########################################################################################

def GACE(path_input, binary_input=False, path_membrane_prefix=None, path_bin=None, path_output=None,
         sigma_membrane=0.9, manual=False, manual_sigma=7, sensitivity=0.99, hard_thresholding=False,
         hard_threshold=1.0,
         sigma_TV=3.6, sigma_LF=0.9, sample=0.2,
         keep_membrane=False, keep_hist=False, keep_all=False, verbose=False):
    '''
    GACE for Global Automated Cell Extractor

    <ENGLISH>

    # Input paths:
    path_input : fused OR binary image that has to be reconstructed
                 /!\ If path_input is a binary image used with the parameter 'binary_input' set to True, then the following conditions must be verified:
                          - path_input = <prefix>.inr[.gz] or <prefix>.<particle>.inr[.gz]
                          - The files <prefix>.theta.inr and <prefix>.phi.inr must exist and must correspond to angle images associated to the orientation of membranes in the binary image
    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths to maxima image (ext) of enhanced images and its associated angle images which give the membranes spatial orientation (theta, phi), in order to avoid their recomputation if possible

    # Output paths:
    path_membrane_prefix+'.[ext|theta|phi].inr' (optional) : paths to save the maxima image (ext) of enhanced images and its associated angle images which give the membranes spatial orientation (theta, phi)
    path_bin (optional) : path to save the binarized membranes image (ie the image sent in input for the tensor voting step)
    path_output (optional) : path to save the output reconstructed image (default is None)

    # Membrane reconstruction parameters
    sigma_membrane=0.9 (default, in real coordinates, adapted to images like Patrick/Ralph/Aquila) : parameter for membranes enhancement filter (before the membranes binarization step)
    sensitivity=0.99 (default) : sensitivity parameter for axial thresholds computation for membranes binarization, with respect to the following criterion
                                 (true positive rate) : threshold = #(membrane class>=threshold)/#(membrane class)

    manual=False (default) : if set to True, this parameter activates the "manual" mode, ie the user fixes the manual parameters for the thresholds computation for membranes binarization
    manual_sigma=7 (default) : manual parameter for the initialization of Rayleigh function sigma parameter for the directional histograms fitting for the thresholds computation

    hard_thresholding=False (default) : This option enables the user to set a "hard threshold" (avoiding the automatic computation of anisotropic thresholds) if set to True.
                                        In that case, the hard_threshold parameter is applied on the whole enhanced image
    hard_threshold=1.0 (default) : if 'hard_thresholding' is set to True, this is the threshold for the binarization of enhanced membranes image
                              (1.0 : adapted for the time-point t001 of Aquila for example)

    sigma_TV=3.6 (default, real coordinates, adapted to images with a spatial resolution of 0.3um^3) : parameter of membranes propagation by the tensor voting algorithm
    sigma_LF=0.9 (default, real coordinates) : parameter for the gaussian blurring of the reconstructed image by the tensor voting method
    sample=0.2 (default) : multiplicative parameter between 0 and 1 that fixes the sampling of voting token from the initial image (1.0 means no resampling) (has an influence on the processing time)


    # Intermediary images keep-or-leave parameters
    keep_membrane=False : if set to True, keeps all the images from membrane enhancement step (extrema and angles images) (is automatically set to True if path_membrane_prefix is provided)
    keep_all=False : if set to True, keeps all the intermediary images generated during the processing


    # Others
    verbose=False : verbosity of the function, displays for example the temporary files generation and deletion



    <FRENCH>

    # Paths d'entree
    path_input : image fusionnee OU image binaire que l'on souhaite reconstruire
                 /!\ Si path_input est une image binaire utilisee avec l'option binary_input=True, veiller a ce que ces conditions soient verifiees :
                          - path_input = <prefix>.inr[.gz] ou <prefix>.<particule>.inr[.gz]
                          - il existe <prefix>.theta.inr et <prefix>.phi.inr qui correspondent aux images d'angles associees a l'orientation des membranes dans l'image binaire

    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths des images de maxima (ext) de membranes rehaussees et de leurs angles associes (theta, phi), afin d'eviter de les recalculer (seulement si binary_input=False)

    # Path de sortie
    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths de sauvegarde des images de maxima (ext) de membranes rehaussees et de leurs angles associes (theta, phi) (seulement si binary_input=False)
    path_bin (optionel) : path de sauvegarde de l'image des membranes binarisees (image envoyee en entree de l'etape de vote de tenseurs)
    path_output (optionel) : path de sauvegarde de l'image reconstruite de sortie (par defaut : None)

    # Membrane reconstruction parameters
    sigma_membrane=0.9 (defaut, unites reelles, adapte a des images de telles que Patrick/Ralph/Aquila) : parametre de rehaussement des
                                                                                   membranes avant binarisation de celles-ci

    sensitivity=0.99 (defaut) : parametre de calcul des seuils anisotropiques selon un critere de sensibilite
                                (true positive rate) : seuil = #(classe membrane>=seuil)/#(classe membrane)
    manual=False (defaut) : parametre activant le mode manuel (ie parametrise) du seuil de binarisation des membranes si egal a True
    manual_sigma=7 (defaut) : parametre manuel d'initialisation pour le calcul du seuil de binarisation des membranes

    hard_thresholding=False (defaut) : Si echec de la precedente methode de seuillage des membranes par calcul automatique de seuils directionnels,
                              possibilite de choisir un seuillage dur global sur l'image en mettant cette option a True
    hard_threshold=1.0 (defaut) : Si hard_thresholding est a True, seuillage des membranes rehaussees via ce seuil
                              (1.0 : adaptee pour le time-point t001 d'Aquila par exemple)

    sigma_TV=3.6 (defaut, reelles, adapte a des images de resolution 0.3um^3) : parametre de propagation des membranes
                                                                                par votes de tenseurs
    sigma_LF=0.9 (defaut, reelles) : parametre de lissage gaussien de l'image des membranes reconstruite
    sample=0.2 (defaut) : echantillonne les votants de l'image initiale selon le coefficient (influe sur la vitesse de traitement)

    # Conserver ou non la transformation non lineaire
    keep_all=False : conserver toutes les images intermediaires servant au calcul
    keep_membrane=False : conserver toutes les images de l'etape de membranes (image d'extrema, d'extrema binarises et d'angles)

    # Divers
    verbose=False : verbosite de la fonction, affiche par exemple les fichiers generes temporairement et ceux qui sont supprimes
    '''

    # Test for input images existence
    # Test existence de l'image d'entree
    assert (os.path.exists(path_input))

    # Parameters for intermediary files keep-or-leave
    # Parametres de conservation des images generees
    if not keep_membrane:
        keep_membrane = keep_all

    # Definition of intermediary image paths
    # Definition des paths d'images intermediaires
    path_WORK = ''
    if path_output:
        path_WORK = os.path.dirname(path_output).rstrip(os.path.sep) + os.path.sep
    else:
        path_WORK = os.path.dirname(path_input).rstrip(os.path.sep) + os.path.sep
    if path_WORK == os.path.sep:
        path_WORK = os.path.curdir + os.path.sep
    tmp_ID = random_number()
    if path_membrane_prefix:
        keep_membrane = True
    else:
        path_membrane_prefix = path_WORK + 'tmp_membrane_' + tmp_ID + ''
    path_TV = path_WORK + 'tmp_reconstructed_' + tmp_ID + '.inr'

    keep_tmp_bin = keep_all
    if not keep_tmp_bin:
        keep_tmp_bin = path_bin and (path_bin != path_membrane_prefix + '.bin.inr')

    if verbose:
        print("Temporary files:")
        if binary_input:
            print(path_TV)
        else:
            if keep_membrane:
                print(path_membrane_prefix + ".bin.inr")
            else:
                print(path_membrane_prefix + ".[bin|ext|theta|phi].inr")
            if not hard_thresholding:
                print(path_membrane_prefix + ".hist.txt")
            print(path_TV)
    if verbose and path_output:
        print("Output file:")
        if path_output:
            print(path_output)

    ### Output path ###
    if not os.path.isdir(path_WORK):
        try:
            os.mkdir(path_WORK)
        except Exception:
            print("Unexpected error: unable to create working directory")

    ### Stuff ###
    keepAll = False
    if keep_all or _instrumented_:
        keepAll = True


    path_tv_input = ''
    if binary_input:
        # Input image = binary image
        path_tv_input = path_input
        keep_membrane = True
    else:
        if (not os.path.exists(path_membrane_prefix + ".ext.inr")) or (
        not os.path.exists(path_membrane_prefix + ".theta.inr")) or (
        not os.path.exists(path_membrane_prefix + ".phi.inr")):
            # Local enhancement of membranes from fused image at t+1 and extraction of the directional maxima (generates the '.[ext|theta|phi].inr')
            # Renforcement des membranes de l'image fusionnee a l'instant t+1 + extraction des maxima directionnels
            cpp_wrapping.obsolete_membrane_renforcement(path_input, prefix_output=path_membrane_prefix, path_mask=None, init=sigma_membrane,
                                  verbose=verbose)

        # Membranes binarization
        if not hard_thresholding:
            # Anisotropic threshold of membranes (the choice of the sensitivity parameter may be critical)
            # Seuillage anisotropique des membranes (parametre de sensitivite potentiellement critique)
            cpp_wrapping.obsolete_anisotropicHist(path_input=path_membrane_prefix + ".ext.inr", path_output=path_membrane_prefix + '.bin.inr',
                            path_mask=None, manual=manual, manual_sigma=manual_sigma, sensitivity=sensitivity,
                            keepAll=keepAll, verbose=verbose)
        else:
            # Hard threshold
            cpp_wrapping.obsolete_seuillage(path_input=path_membrane_prefix + ".ext.inr", path_output=path_membrane_prefix + '.bin.inr',
                      sb=hard_threshold, verbose=verbose)
        path_tv_input = path_membrane_prefix + ".bin.inr"
        if path_bin and not os.path.exists(path_bin):
            # Copy of the temporary binary image to the path provided in parameter
            # Copie de l'image binaire temporaire vers le path renseigne en parametre
            assert os.path.exists(path_membrane_prefix + '.bin.inr')
            cpp_wrapping.obsolete_copy(path_membrane_prefix + ".bin.inr", path_bin, verbose=verbose)

    # Tensor voting on the image of binarized membranes
    if verbose:
        print('Processing Tensor Voting on image ' + path_tv_input + ' ...')
    assert (os.path.exists(path_tv_input))
    assert (path_tv_input.endswith('.inr.gz') or path_tv_input.endswith('.inr'))
    binary_file_decomp = path_tv_input.split('.')
    if binary_file_decomp[-1] == 'gz':
        binary_file_decomp = binary_file_decomp[:-1]
    assert binary_file_decomp[-1] == 'inr'
    binary_file_decomp = binary_file_decomp[:-1]
    if (not os.path.exists('.'.join(binary_file_decomp) + '.theta.inr')) or (
    not os.path.exists('.'.join(binary_file_decomp) + '.phi.inr')):
        assert (len(binary_file_decomp) > 1 and os.path.exists(
            '.'.join(binary_file_decomp[:-1]) + '.theta.inr') and os.path.exists('.'.join(binary_file_decomp[
                                                                                          :-1]) + '.phi.inr')), "Error : unexpectedly, <prefix>.theta.inr and/or <prefix>.phi.inr not found for file " + path_tv_input + " before tensor voting step"

    cpp_wrapping.obsolete_TVmembrane(path_input=path_tv_input, path_output=path_TV, path_mask=None, scale=sigma_TV, sample=sample,
               sigma_LF=sigma_LF, realScale=True, keepAll=keepAll, verbose=verbose)

    # Reading of the reconstructed image (the one returned by the function)
    reconstructed_image = imread(path_TV)

    # Copy to the output image to the provided output path
    if path_output and path_TV != path_output:
        cpp_wrapping.obsolete_copy(path_TV, path_output, verbose=verbose)

    # Deletion of intermediary images
    files_to_rm = ""

    if not keep_all:
        if path_TV and path_TV != path_output and os.path.exists(path_TV):
            files_to_rm += path_TV + " "

    if not binary_input and not keep_membrane:
        if path_membrane_prefix:
            path_membrane_dir = os.path.dirname(path_membrane_prefix)
            files_to_rm += ''.join([path_membrane_dir + os.path.sep + i + " " for i in os.listdir(path_membrane_dir) if
                                    os.path.isfile(path_membrane_dir + os.path.sep + i) and (
                                                path_membrane_dir + os.path.sep + i).count(path_membrane_prefix)])

    if not keep_membrane:
        if path_membrane_prefix:
            if os.path.exists(path_membrane_prefix + '.ext.inr'):
                files_to_rm += path_membrane_prefix + '.ext.inr' + " "
            if os.path.exists(path_membrane_prefix + '.theta.inr'):
                files_to_rm += path_membrane_prefix + '.theta.inr' + " "
            if os.path.exists(path_membrane_prefix + '.phi.inr'):
                files_to_rm += path_membrane_prefix + '.phi.inr' + " "

    if not keep_tmp_bin and path_membrane_prefix and os.path.exists(path_membrane_prefix + '.bin.inr'):
        files_to_rm += path_membrane_prefix + '.bin.inr' + " "

    if not keep_hist and path_membrane_prefix and os.path.exists(path_membrane_prefix + '.hist.txt'):
        files_to_rm += path_membrane_prefix + '.hist.txt' + " "

    if files_to_rm and not _instrumented_:
        if verbose:
            print("Deleting temporary files: \n" + files_to_rm)
        os.system("rm " + files_to_rm)

    # End of GACE
    return reconstructed_image


def _GACE(path_input, binary_input=False, path_membrane_prefix=None, path_bin=None, path_output=None,
         sigma_membrane=0.9, manual=False, manual_sigma=7, sensitivity=0.99, hard_thresholding=False,
         hard_threshold=1.0,
         sigma_TV=3.6, sigma_LF=0.9, sample=0.2,
         keep_membrane=False, keep_hist=False, keep_all=False, verbose=False):
    '''
    GACE for Global Automated Cell Extractor

    <ENGLISH>

    # Input paths:
    path_input : fused OR binary image that has to be reconstructed
                 /!\ If path_input is a binary image used with the parameter 'binary_input' set to True, then the following conditions must be verified:
                          - path_input = <prefix>.inr[.gz] or <prefix>.<particle>.inr[.gz]
                          - The files <prefix>.theta.inr and <prefix>.phi.inr must exist and must correspond to angle images associated to the orientation of membranes in the binary image
    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths to maxima image (ext) of enhanced images and its associated angle images which give the membranes spatial orientation (theta, phi), in order to avoid their recomputation if possible

    # Output paths:
    path_membrane_prefix+'.[ext|theta|phi].inr' (optional) : paths to save the maxima image (ext) of enhanced images and its associated angle images which give the membranes spatial orientation (theta, phi)
    path_bin (optional) : path to save the binarized membranes image (ie the image sent in input for the tensor voting step)
    path_output (optional) : path to save the output reconstructed image (default is None)

    # Membrane reconstruction parameters
    sigma_membrane=0.9 (default, in real coordinates, adapted to images like Patrick/Ralph/Aquila) : parameter for membranes enhancement filter (before the membranes binarization step)
    sensitivity=0.99 (default) : sensitivity parameter for axial thresholds computation for membranes binarization, with respect to the following criterion
                                 (true positive rate) : threshold = #(membrane class>=threshold)/#(membrane class)

    manual=False (default) : if set to True, this parameter activates the "manual" mode, ie the user fixes the manual parameters for the thresholds computation for membranes binarization
    manual_sigma=7 (default) : manual parameter for the initialization of Rayleigh function sigma parameter for the directional histograms fitting for the thresholds computation

    hard_thresholding=False (default) : This option enables the user to set a "hard threshold" (avoiding the automatic computation of anisotropic thresholds) if set to True.
                                        In that case, the hard_threshold parameter is applied on the whole enhanced image
    hard_threshold=1.0 (default) : if 'hard_thresholding' is set to True, this is the threshold for the binarization of enhanced membranes image
                              (1.0 : adapted for the time-point t001 of Aquila for example)

    sigma_TV=3.6 (default, real coordinates, adapted to images with a spatial resolution of 0.3um^3) : parameter of membranes propagation by the tensor voting algorithm
    sigma_LF=0.9 (default, real coordinates) : parameter for the gaussian blurring of the reconstructed image by the tensor voting method
    sample=0.2 (default) : multiplicative parameter between 0 and 1 that fixes the sampling of voting token from the initial image (1.0 means no resampling) (has an influence on the processing time)


    # Intermediary images keep-or-leave parameters
    keep_membrane=False : if set to True, keeps all the images from membrane enhancement step (extrema and angles images) (is automatically set to True if path_membrane_prefix is provided)
    keep_all=False : if set to True, keeps all the intermediary images generated during the processing


    # Others
    verbose=False : verbosity of the function, displays for example the temporary files generation and deletion



    <FRENCH>

    # Paths d'entree
    path_input : image fusionnee OU image binaire que l'on souhaite reconstruire
                 /!\ Si path_input est une image binaire utilisee avec l'option binary_input=True, veiller a ce que ces conditions soient verifiees :
                          - path_input = <prefix>.inr[.gz] ou <prefix>.<particule>.inr[.gz]
                          - il existe <prefix>.theta.inr et <prefix>.phi.inr qui correspondent aux images d'angles associees a l'orientation des membranes dans l'image binaire

    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths des images de maxima (ext) de membranes rehaussees et de leurs angles associes (theta, phi), afin d'eviter de les recalculer (seulement si binary_input=False)

    # Path de sortie
    path_membrane_prefix+'.[ext|theta|phi].inr' (optionel) : paths de sauvegarde des images de maxima (ext) de membranes rehaussees et de leurs angles associes (theta, phi) (seulement si binary_input=False)
    path_bin (optionel) : path de sauvegarde de l'image des membranes binarisees (image envoyee en entree de l'etape de vote de tenseurs)
    path_output (optionel) : path de sauvegarde de l'image reconstruite de sortie (par defaut : None)

    # Membrane reconstruction parameters
    sigma_membrane=0.9 (defaut, unites reelles, adapte a des images de telles que Patrick/Ralph/Aquila) : parametre de rehaussement des
                                                                                   membranes avant binarisation de celles-ci

    sensitivity=0.99 (defaut) : parametre de calcul des seuils anisotropiques selon un critere de sensibilite
                                (true positive rate) : seuil = #(classe membrane>=seuil)/#(classe membrane)
    manual=False (defaut) : parametre activant le mode manuel (ie parametrise) du seuil de binarisation des membranes si egal a True
    manual_sigma=7 (defaut) : parametre manuel d'initialisation pour le calcul du seuil de binarisation des membranes

    hard_thresholding=False (defaut) : Si echec de la precedente methode de seuillage des membranes par calcul automatique de seuils directionnels,
                              possibilite de choisir un seuillage dur global sur l'image en mettant cette option a True
    hard_threshold=1.0 (defaut) : Si hard_thresholding est a True, seuillage des membranes rehaussees via ce seuil
                              (1.0 : adaptee pour le time-point t001 d'Aquila par exemple)

    sigma_TV=3.6 (defaut, reelles, adapte a des images de resolution 0.3um^3) : parametre de propagation des membranes
                                                                                par votes de tenseurs
    sigma_LF=0.9 (defaut, reelles) : parametre de lissage gaussien de l'image des membranes reconstruite
    sample=0.2 (defaut) : echantillonne les votants de l'image initiale selon le coefficient (influe sur la vitesse de traitement)

    # Conserver ou non la transformation non lineaire
    keep_all=False : conserver toutes les images intermediaires servant au calcul
    keep_membrane=False : conserver toutes les images de l'etape de membranes (image d'extrema, d'extrema binarises et d'angles)

    # Divers
    verbose=False : verbosite de la fonction, affiche par exemple les fichiers generes temporairement et ceux qui sont supprimes
    '''

    # Test for input images existence
    # Test existence de l'image d'entree
    assert (os.path.exists(path_input))

    # Parameters for intermediary files keep-or-leave
    # Parametres de conservation des images generees
    if not keep_membrane:
        keep_membrane = keep_all

    # Definition of intermediary image paths
    # Definition des paths d'images intermediaires
    path_WORK = ''
    if path_output:
        path_WORK = os.path.dirname(path_output).rstrip(os.path.sep) + os.path.sep
    else:
        path_WORK = os.path.dirname(path_input).rstrip(os.path.sep) + os.path.sep
    if path_WORK == os.path.sep:
        path_WORK = os.path.curdir + os.path.sep
    tmp_ID = random_number()
    if path_membrane_prefix:
        keep_membrane = True
    else:
        path_membrane_prefix = path_WORK + 'tmp_membrane_' + tmp_ID + ''
    path_TV = path_WORK + 'tmp_reconstructed_' + tmp_ID + '.inr'

    keep_tmp_bin = keep_all
    if not keep_tmp_bin:
        keep_tmp_bin = path_bin and (path_bin != path_membrane_prefix + '.bin.inr')

    if verbose:
        print("Temporary files:")
        if binary_input:
            print(path_TV)
        else:
            if keep_membrane:
                print(path_membrane_prefix + ".bin.inr")
            else:
                print(path_membrane_prefix + ".[bin|ext|theta|phi].inr")
            if not hard_thresholding:
                print(path_membrane_prefix + ".hist.txt")
            print(path_TV)
    if verbose and path_output:
        print("Output file:")
        if path_output:
            print(path_output)

    ### Output path ###
    if not os.path.isdir(path_WORK):
        try:
            os.mkdir(path_WORK)
        except Exception:
            print("Unexpected error: unable to create working directory")

    ### Stuff ###

    path_tv_input = ''
    if binary_input:
        # Input image = binary image
        path_tv_input = path_input
        keep_membrane = True
    else:
        if (not os.path.exists(path_membrane_prefix + ".ext.inr")) or (
        not os.path.exists(path_membrane_prefix + ".theta.inr")) or (
        not os.path.exists(path_membrane_prefix + ".phi.inr")):
            # Local enhancement of membranes from fused image at t+1 and extraction of the directional maxima (generates the '.[ext|theta|phi].inr')
            # Renforcement des membranes de l'image fusionnee a l'instant t+1 + extraction des maxima directionnels
            membrane_renforcement(path_input, prefix_output=path_membrane_prefix, path_mask=None, init=sigma_membrane,
                                  verbose=verbose)

        # Membranes binarization
        if not hard_thresholding:
            # Anisotropic threshold of membranes (the choice of the sensitivity parameter may be critical)
            # Seuillage anisotropique des membranes (parametre de sensitivite potentiellement critique)
            anisotropicHist(path_input=path_membrane_prefix + ".ext.inr", path_output=path_membrane_prefix + '.bin.inr',
                            path_mask=None, manual=manual, manual_sigma=manual_sigma, sensitivity=sensitivity,
                            keepAll=False, verbose=verbose)
        else:
            # Hard threshold
            seuillage(path_input=path_membrane_prefix + ".ext.inr", path_output=path_membrane_prefix + '.bin.inr',
                      sb=hard_threshold, verbose=verbose)
        path_tv_input = path_membrane_prefix + ".bin.inr"
        if path_bin and not os.path.exists(path_bin):
            # Copy of the temporary binary image to the path provided in parameter
            # Copie de l'image binaire temporaire vers le path renseigne en parametre
            assert os.path.exists(path_membrane_prefix + '.bin.inr')
            copy(path_membrane_prefix + ".bin.inr", path_bin, verbose=verbose)

    # Tensor voting on the image of binarized membranes
    if verbose:
        print('Processing Tensor Voting on image ' + path_tv_input + ' ...')
    assert (os.path.exists(path_tv_input))
    assert (path_tv_input.endswith('.inr.gz') or path_tv_input.endswith('.inr'))
    binary_file_decomp = path_tv_input.split('.')
    if binary_file_decomp[-1] == 'gz':
        binary_file_decomp = binary_file_decomp[:-1]
    assert binary_file_decomp[-1] == 'inr'
    binary_file_decomp = binary_file_decomp[:-1]
    if (not os.path.exists('.'.join(binary_file_decomp) + '.theta.inr')) or (
    not os.path.exists('.'.join(binary_file_decomp) + '.phi.inr')):
        assert (len(binary_file_decomp) > 1 and os.path.exists(
            '.'.join(binary_file_decomp[:-1]) + '.theta.inr') and os.path.exists('.'.join(binary_file_decomp[
                                                                                          :-1]) + '.phi.inr')), "Error : unexpectedly, <prefix>.theta.inr and/or <prefix>.phi.inr not found for file " + path_tv_input + " before tensor voting step"

    TVmembrane(path_input=path_tv_input, path_output=path_TV, path_mask=None, scale=sigma_TV, sample=sample,
               sigma_LF=sigma_LF, realScale=True, keepAll=False, verbose=verbose)

    # Reading of the reconstructed image (the one returned by the function)
    reconstructed_image = imread(path_TV)

    # Copy to the output image to the provided output path
    if path_output and path_TV != path_output:
        copy(path_TV, path_output, verbose=verbose)

    # Deletion of intermediary images
    files_to_rm = ""

    if not keep_all:
        if path_TV and path_TV != path_output and os.path.exists(path_TV):
            files_to_rm += path_TV + " "

    if not binary_input and not keep_membrane:
        if path_membrane_prefix:
            path_membrane_dir = os.path.dirname(path_membrane_prefix)
            files_to_rm += ''.join([path_membrane_dir + os.path.sep + i + " " for i in os.listdir(path_membrane_dir) if
                                    os.path.isfile(path_membrane_dir + os.path.sep + i) and (
                                                path_membrane_dir + os.path.sep + i).count(path_membrane_prefix)])

    if not keep_membrane:
        if path_membrane_prefix:
            if os.path.exists(path_membrane_prefix + '.ext.inr'):
                files_to_rm += path_membrane_prefix + '.ext.inr' + " "
            if os.path.exists(path_membrane_prefix + '.theta.inr'):
                files_to_rm += path_membrane_prefix + '.theta.inr' + " "
            if os.path.exists(path_membrane_prefix + '.phi.inr'):
                files_to_rm += path_membrane_prefix + '.phi.inr' + " "

    if not keep_tmp_bin and path_membrane_prefix and os.path.exists(path_membrane_prefix + '.bin.inr'):
        files_to_rm += path_membrane_prefix + '.bin.inr' + " "

    if not keep_hist and path_membrane_prefix and os.path.exists(path_membrane_prefix + '.hist.txt'):
        files_to_rm += path_membrane_prefix + '.hist.txt' + " "

    if files_to_rm and not _instrumented_:
        if verbose:
            print("Deleting temporary files: \n" + files_to_rm)
        os.system("rm " + files_to_rm)

    # End of GACE
    return reconstructed_image

