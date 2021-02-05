
import os
import imp
import sys

import ace
import common
import CommunFunctions.cpp_wrapping as cpp_wrapping

monitoring = common.Monitoring()


########################################################################################
#
#
#
########################################################################################


class ReconstructionParameters(ace.AceParameters):

    ############################################################
    #
    # initialisation
    #
    ############################################################

    def __init__(self, prefix=None):

        ace.AceParameters.__init__(self, prefix=prefix)

        #
        #
        #
        self.intensity_transformation = 'Identity'
        self.intensity_enhancement = None
        self.outer_contour_enhancement = False

        self.reconstruction_images_combination = "maximum"

        self.cell_normalization_min_method = 'cellinterior'
        self.cell_normalization_max_method = 'cellborder'

        self.normalization_min_percentile = 0.01
        self.normalization_max_percentile = 0.99
        self.cell_normalization_sigma = 5.0

        #
        # registration parameters
        #
        self.registration = []

        self.registration.append(common.RegistrationParameters(prefix=[self._prefix, 'linear_registration_']))
        self.registration[0].pyramid_highest_level = 5
        self.registration[0].pyramid_lowest_level = 3
        self.registration[0].gaussian_pyramid = True
        self.registration[0].transformation_type = 'affine'
        self.registration[0].transformation_estimation_type = 'wlts'
        self.registration[0].lts_fraction = 0.55

        self.registration.append(common.RegistrationParameters(prefix=[self._prefix, 'nonlinear_registration_']))
        self.registration[1].pyramid_highest_level = 5
        self.registration[1].pyramid_lowest_level = 3
        self.registration[1].gaussian_pyramid = True
        self.registration[1].transformation_type = 'vectorfield'
        self.registration[1].elastic_sigma = 2.0
        self.registration[1].transformation_estimation_type = 'wlts'
        self.registration[1].lts_fraction = 1.0
        self.registration[1].fluid_sigma = 2.0
        #
        #
        #
        self.keep_reconstruction = True

    ############################################################
    #
    # print / write
    #
    ############################################################

    def print_parameters(self):
        print("")
        print('#')
        print('# ReconstructionParameters')
        print('#')

        common.PrefixedParameter.print_parameters(self)

        ace.AceParameters.print_parameters(self)
        self.varprint('intensity_transformation', self.intensity_transformation)
        self.varprint('intensity_enhancement', self.intensity_enhancement)
        self.varprint('outer_contour_enhancement', self.outer_contour_enhancement)
        self.varprint('reconstruction_images_combination', self.reconstruction_images_combination)

        self.varprint('cell_normalization_min_method', self.cell_normalization_min_method)
        self.varprint('cell_normalization_max_method', self.cell_normalization_max_method)

        self.varprint('normalization_min_percentile', self.normalization_min_percentile)
        self.varprint('normalization_max_percentile', self.normalization_max_percentile)
        self.varprint('cell_normalization_sigma', self.cell_normalization_sigma)

        for p in self.registration:
            p.print_parameters()

        self.varprint('keep_reconstruction', self.keep_reconstruction)
        print("")

    def write_parameters_in_file(self, logfile):
        logfile.write("\n")
        logfile.write("# \n")
        logfile.write("# ReconstructionParameters\n")
        logfile.write("# \n")

        common.PrefixedParameter.write_parameters_in_file(self, logfile)

        ace.AceParameters.write_parameters_in_file(self, logfile)

        doc = "possible values are 'identity', 'normalization_to_u8',"
        doc += "'cell_normalization_to_u8', 'none' or None"
        self.varwrite(logfile, 'intensity_transformation', self.intensity_transformation, doc)
        doc = "possible values are 'gace', 'glace', 'none', or None\n"
        doc += "glace is a cell-based enhancement technique, only available in the\n"
        doc += "segmentation propagtion step (aka astec)"
        self.varwrite(logfile, 'intensity_enhancement', self.intensity_enhancement, doc)
        self.varwrite(logfile, 'outer_contour_enhancement', self.outer_contour_enhancement)
        self.varwrite(logfile, 'reconstruction_images_combination', self.reconstruction_images_combination)

        self.varwrite(logfile, 'cell_normalization_min_method', self.cell_normalization_min_method)
        self.varwrite(logfile, 'cell_normalization_max_method', self.cell_normalization_max_method)

        self.varwrite(logfile, 'normalization_min_percentile', self.normalization_min_percentile)
        self.varwrite(logfile, 'normalization_max_percentile', self.normalization_max_percentile)
        self.varwrite(logfile, 'cell_normalization_sigma', self.cell_normalization_sigma)

        for p in self.registration:
            p.write_parameters_in_file(logfile)

        self.varwrite(logfile, 'keep_reconstruction', self.keep_reconstruction)

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

    def update_from_parameters(self, parameters):
        ace.AceParameters.update_from_parameters(self, parameters)

        #
        # reconstruction method
        #

        self.intensity_transformation = self.read_parameter(parameters, 'intensity_transformation',
                                                            self.intensity_transformation)
        self.intensity_enhancement = self.read_parameter(parameters, 'intensity_enhancement',
                                                         self.intensity_enhancement)
        self.outer_contour_enhancement = self.read_parameter(parameters, 'outer_contour_enhancement',
                                                         self.outer_contour_enhancement)
        self.reconstruction_images_combination = self.read_parameter(parameters, 'reconstruction_images_combination',
                                                         self.reconstruction_images_combination)

        self.cell_normalization_min_method = self.read_parameter(parameters, 'cell_normalization_min_method',
                                                                 self.cell_normalization_min_method)
        self.cell_normalization_max_method = self.read_parameter(parameters, 'cell_normalization_max_method',
                                                                 self.cell_normalization_max_method)
        self.normalization_min_percentile = self.read_parameter(parameters, 'normalization_min_percentile',
                                                                self.normalization_min_percentile)
        self.normalization_max_percentile = self.read_parameter(parameters, 'normalization_max_percentile',
                                                                self.normalization_max_percentile)
        self.cell_normalization_sigma = self.read_parameter(parameters, 'cell_normalization_sigma',
                                                            self.cell_normalization_sigma)

        #
        #
        #
        self.registration[0].update_from_parameters(parameters)
        self.registration[1].update_from_parameters(parameters)

        #
        #
        #
        self.keep_reconstruction = self.read_parameter(parameters, 'keep_reconstruction', self.keep_reconstruction)

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
        proc = "is_equal"
        _trace_ = False
        if self.intensity_transformation != p.intensity_transformation:
            if _trace_:
                print(proc + "not equal at 'intensity_transformation'")
            return False
        if self.intensity_enhancement != p.intensity_enhancement:
            if _trace_:
                print(proc + "not equal at 'intensity_enhancement'")
            return False
        if self.outer_contour_enhancement != p.outer_contour_enhancement:
            if _trace_:
                print(proc + "not equal at 'outer_contour_enhancement'")
            return False
        if self.reconstruction_images_combination != p.reconstruction_images_combination:
            if _trace_:
                print(proc + "not equal at 'reconstruction_images_combination'")
            return False
        if self.cell_normalization_min_method != p.cell_normalization_min_method:
            if _trace_:
                print(proc + "not equal at 'cell_normalization_min_method'")
            return False
        if self.cell_normalization_max_method != p.cell_normalization_max_method:
            if _trace_:
                print(proc + "not equal at 'cell_normalization_max_method'")
            return False
        if self.normalization_min_percentile != p.normalization_min_percentile:
            if _trace_:
                print(proc + "not equal at 'normalization_min_percentile'")
            return False
        if self.normalization_max_percentile != p.normalization_max_percentile:
            if _trace_:
                print(proc + "not equal at 'normalization_max_percentile'")
            return False
        if self.cell_normalization_sigma != p.cell_normalization_sigma:
            if _trace_:
                print(proc + "not equal at 'cell_normalization_sigma'")
            return False

        if ace.AceParameters.is_equal(self, p) is False:
            if _trace_:
                print(proc + "not equal at 'AceParameters'")
            return False

        return True

#
#
#
#
#

def get_deformation_from_current_to_previous(current_time, experiment, parameters, previous_time):
    """

    :param current_time:
    :param experiment:
    :param parameters:
    :param previous_time:
    :return:
    """

    proc = 'get_deformation_from_current_to_previous'

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, ReconstructionParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    # image to be registered
    # floating image = image at previous time
    # reference image = image at current time
    #

    current_name = experiment.fusion_dir.get_image_name(current_time)
    current_image = common.find_file(experiment.fusion_dir.get_directory(), current_name, file_type='image',
                                     callfrom=proc, local_monitoring=None, verbose=False)
    if current_image is None:
        monitoring.to_log_and_console("    .. " + proc + " no fused image was found for time " + str(current_time), 2)
        return None
    current_image = os.path.join(experiment.fusion_dir.get_directory(), current_image)

    previous_name = experiment.fusion_dir.get_image_name(previous_time)
    previous_image = common.find_file(experiment.fusion_dir.get_directory(), previous_name, file_type='image',
                                      callfrom=proc, local_monitoring=None, verbose=False)
    if previous_image is None:
        monitoring.to_log_and_console("    .. " + proc + " no fused image was found for time " + str(previous_time), 2)
        return None
    previous_image = os.path.join(experiment.fusion_dir.get_directory(), previous_image)

    #
    # auxiliary file names
    #
    affine_image = common.add_suffix(previous_image, "_affine", new_dirname=experiment.working_dir.get_tmp_directory(0),
                                     new_extension=experiment.default_image_suffix)
    affine_trsf = common.add_suffix(previous_image, "_affine", new_dirname=experiment.working_dir.get_tmp_directory(0),
                                    new_extension="trsf")
    vector_image = common.add_suffix(previous_image, "_vector", new_dirname=experiment.working_dir.get_tmp_directory(0),
                                     new_extension=experiment.default_image_suffix)
    vector_trsf = common.add_suffix(previous_image, "_vector", new_dirname=experiment.working_dir.get_tmp_directory(0),
                                    new_extension="trsf")

    if os.path.isfile(vector_trsf) and not monitoring.forceResultsToBeBuilt:
        return vector_trsf

    #
    # linear registration
    #
    common.blockmatching(current_image, previous_image, affine_image, affine_trsf, None, parameters.registration[0])

    if not os.path.isfile(affine_image) or not os.path.isfile(affine_trsf):
        monitoring.to_log_and_console("    .. " + proc + " linear registration failed for time " + str(current_time), 2)
        return None

    #
    # non-linear registration
    #
    common.blockmatching(current_image, previous_image, vector_image, vector_trsf, affine_trsf,
                         parameters.registration[1])
    if not os.path.isfile(vector_image) or not os.path.isfile(vector_trsf):
        monitoring.to_log_and_console("    .. " + proc + " non-linear registration failed for time " +
                                      str(current_time), 2)
        return None

    return vector_trsf

#
#
#
#
#


def get_previous_deformed_segmentation(current_time, experiment, parameters, previous_time=None):
    """

    :param current_time:
    :param experiment:
    :param parameters:
    :param previous_time:
    :return:
    """

    #
    #
    #

    proc = 'get_previous_deformed_segmentation'

    #
    # it will generate (in the temporary_path directory)
    #
    # files related to the deformation computation
    # - $EN_fuse_t$TIME_affine.inr
    # - $EN_fuse_t$TIME_affine.trsf
    # - $EN_fuse_t$TIME_vector.inr
    # - $EN_fuse_t$TIME_vector.trsf
    #
    # the deformed segmentation
    # - $EN_[mars,seg]_t$TIME_deformed.inr
    #

    prev_segimage = experiment.get_segmentation_image(previous_time)
    if prev_segimage is None:
        if previous_time is None:
            monitoring.to_log_and_console("    .. " + proc + ": was called with 'previous_time = None'", 2)
        else:
            monitoring.to_log_and_console("    .. " + proc + ": no segmentation image was found for time "
                                          + str(previous_time), 2)
        return None

    prev_def_segimage = common.add_suffix(prev_segimage, "_deformed",
                                          new_dirname=experiment.working_dir.get_tmp_directory(0),
                                          new_extension=experiment.default_image_suffix)

    if os.path.isfile(prev_def_segimage):
        return prev_def_segimage

    deformation = get_deformation_from_current_to_previous(current_time, experiment, parameters, previous_time)

    if deformation is None:
        monitoring.to_log_and_console("    .. " + proc + ": no deformation was found for time "
                                      + str(current_time) + " towards " + str(previous_time), 2)
        return None

    monitoring.to_log_and_console("    .. resampling of '" + str(prev_segimage).split(os.path.sep)[-1] + "'", 2)

    cpp_wrapping.apply_transformation(prev_segimage, prev_def_segimage, deformation,
                                      interpolation_mode='nearest', monitoring=monitoring)

    return prev_def_segimage


########################################################################################
#
#
#
########################################################################################


def build_reconstructed_image(current_time, experiment, parameters, suffix=None, previous_time=None):
    """

    :param current_time:
    :param experiment:
    :param parameters:
    :param suffix:
    :param previous_time:
    :return:
    """
    #
    # build input image(s) for segmentation
    # 0. build names
    # 1. do image enhancement if required
    # 2. transform input image if required
    # 3. mix the two results
    #
    # If any transformation is performed on the input image, the final result image (input of the watershed)
    # will be named (in the 'temporary_path' directory)
    # - 'input_image'_membrane
    # if fusion is required
    # - 'input_image'_enhanced
    #   is the membrance-enhanced image (by tensor voting method)
    # - 'input_image'_intensity
    #   is the intensity-transformed image (by normalization)
    #

    proc = "build_reconstructed_image"
    _trace_ = True
    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, ReconstructionParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    if suffix is None:
        local_suffix = "reconstructed_"
    else:
        local_suffix = suffix

    #
    # get fused image
    #
    input_dir = experiment.fusion_dir.get_directory(0)
    input_name = experiment.fusion_dir.get_image_name(current_time)

    input_image = common.find_file(input_dir, input_name, file_type='image', callfrom=proc, local_monitoring=monitoring)

    if input_image is None:
        monitoring.to_log_and_console("    .. image '" + input_name + "' not found in '" + str(input_dir) + "'", 2)
        return None
    input_image = os.path.join(input_dir, input_image)

    #
    # set image names
    #
    # there can be intensity enhancement (ie filtering based membrane enhancement)
    #
    # if no intensity (i.e. membrane) enhancement
    #    if no intensity
    #
    #

    intensity_image = None
    enhanced_image = None
    contour_image = None
    reconstructed_image = common.add_suffix(input_image, local_suffix,
                                            new_dirname=experiment.working_dir.get_rec_directory(0),
                                            new_extension=experiment.default_image_suffix)

    intensity_image_bytes = 0
    enhanced_image_bytes = 0
    contour_image_bytes = 0

    #
    # there is a histogram-based intensity transformation
    # intensity_image has to be set
    #
    if not (parameters.intensity_transformation ==  None or parameters.intensity_transformation.lower() == 'none'):
        if parameters.intensity_transformation.lower() == 'identity':
            intensity_image = input_image
            intensity_image_bytes = 2
        elif parameters.intensity_transformation.lower() == 'normalization_to_u8' \
                or parameters.intensity_transformation.lower() == 'global_normalization_to_u8' \
                or parameters.intensity_transformation.lower() == 'cell_normalization_to_u8':
            #
            # set the 'intensity_image' name
            #
            if (parameters.intensity_enhancement is None or parameters.intensity_enhancement.lower() == 'none') \
                    and parameters.outer_contour_enhancement is False:
                #
                # there is only the histogram-based intensity transformation
                #
                experiment.working_dir.make_rec_directory()
                intensity_image = reconstructed_image
            else:
                #
                # there is an other operation
                #
                intensity_image = common.add_suffix(input_image, local_suffix + "_intensity",
                                                    new_dirname=experiment.working_dir.get_tmp_directory(0),
                                                    new_extension=experiment.default_image_suffix)
            intensity_image_bytes = 1
            #
            # compute 'intensity_image'
            #
            if parameters.intensity_transformation.lower() == 'normalization_to_u8' \
                    or parameters.intensity_transformation.lower() == 'global_normalization_to_u8':
                if (not os.path.isfile(intensity_image) and (reconstructed_image is None
                                                             or not os.path.isfile(reconstructed_image))) \
                        or monitoring.forceResultsToBeBuilt is True:
                    monitoring.to_log_and_console("    .. intensity global normalization of '"
                                                  + str(input_image).split(os.path.sep)[-1] + "'", 2)
                    cpp_wrapping.global_normalization_to_u8(input_image, intensity_image,
                                                            min_percentile=parameters.normalization_min_percentile,
                                                            max_percentile=parameters.normalization_max_percentile,
                                                            other_options=None, monitoring=monitoring)
            elif parameters.intensity_transformation.lower() == 'cell_normalization_to_u8':
                if (not os.path.isfile(intensity_image) and (reconstructed_image is None
                                                             or not os.path.isfile(reconstructed_image))) \
                        or monitoring.forceResultsToBeBuilt is True:
                    monitoring.to_log_and_console("    .. intensity cell-based normalization of '"
                                                  + str(input_image).split(os.path.sep)[-1] + "'", 2)
                    if previous_time is None:
                        monitoring.to_log_and_console("       previous time point was not given", 2)
                        monitoring.to_log_and_console("    .. " + proc + ": switch to 'global normalization' ", 1)
                        cpp_wrapping.global_normalization_to_u8(input_image, intensity_image,
                                                                min_percentile=parameters.normalization_min_percentile,
                                                                max_percentile=parameters.normalization_max_percentile,
                                                                other_options=None, monitoring=monitoring)
                    else:
                        previous_deformed_segmentation = get_previous_deformed_segmentation(current_time, experiment,
                                                                                            parameters, previous_time)
                        cpp_wrapping.cell_normalization_to_u8(input_image, previous_deformed_segmentation,
                                                              intensity_image,
                                                              min_percentile=parameters.normalization_min_percentile,
                                                              max_percentile=parameters.normalization_max_percentile,
                                                              cell_normalization_min_method=parameters.cell_normalization_min_method,
                                                              cell_normalization_max_method=parameters.cell_normalization_max_method,
                                                              sigma=parameters.cell_normalization_sigma,
                                                              other_options=None, monitoring=monitoring)
        else:
            monitoring.to_log_and_console("    unknown intensity transformation method: '"
                                          + str(parameters.intensity_transformation) + "'", 2)
        #
        # no fusion to be done
        #
        if (parameters.intensity_enhancement is None or parameters.intensity_enhancement.lower() == 'none') \
                    and parameters.outer_contour_enhancement is False:
            return intensity_image

    #
    # there is a membrane enhancement image
    # enhanced_image has to be set
    #
    if not (parameters.intensity_enhancement == None or parameters.intensity_enhancement.lower() == 'none'):
        if parameters.intensity_enhancement.lower() == 'gace' or parameters.intensity_enhancement.lower() == 'glace':
            #
            # set the 'enhanced_image' name
            #
            if (parameters.intensity_transformation is None or parameters.intensity_transformation.lower() == 'none') \
                    and parameters.outer_contour_enhancement is False:
                #
                # there is only the membrane enhancement
                #
                experiment.working_dir.make_rec_directory()
                enhanced_image = reconstructed_image
            else:
                #
                # there is an other operation
                #
                enhanced_image = common.add_suffix(input_image, local_suffix + "_enhanced",
                                                    new_dirname=experiment.working_dir.get_tmp_directory(0),
                                                    new_extension=experiment.default_image_suffix)
            enhanced_image_bytes = 1
            #
            # compute 'enhanced_image'
            #
            if parameters.intensity_enhancement.lower() == 'gace':
                if (not os.path.isfile(enhanced_image) and (reconstructed_image is None
                                                            or not os.path.isfile(reconstructed_image))) \
                        or monitoring.forceResultsToBeBuilt is True:
                    monitoring.to_log_and_console("    .. global enhancement of '"
                                                  + str(input_image).split(os.path.sep)[-1] + "'", 2)
                    ace.monitoring.copy(monitoring)
                    ace.global_membrane_enhancement(input_image, enhanced_image, experiment,
                                                    temporary_path=experiment.working_dir.get_tmp_directory(0),
                                                    parameters=parameters)
            elif parameters.intensity_enhancement.lower() == 'glace':
                if (not os.path.isfile(enhanced_image) and (reconstructed_image is None
                                                            or not os.path.isfile(reconstructed_image))) \
                        or monitoring.forceResultsToBeBuilt is True:
                    monitoring.to_log_and_console("    .. cell enhancement of '"
                                                  + str(input_image).split(os.path.sep)[-1] + "'", 2)
                    ace.monitoring.copy(monitoring)
                    previous_deformed_segmentation = get_previous_deformed_segmentation(current_time, experiment,
                                                                                        parameters,
                                                                                        previous_time)
                    if previous_deformed_segmentation is None:
                        monitoring.to_log_and_console("    .. " + proc + ": switch to 'ace' ", 1)
                        ace.global_membrane_enhancement(input_image, enhanced_image, experiment,
                                                        temporary_path=experiment.working_dir.get_tmp_directory(0),
                                                        parameters=parameters)
                    else:
                        ace.cell_membrane_enhancement(input_image, previous_deformed_segmentation, enhanced_image,
                                                      experiment,
                                                      temporary_path=experiment.working_dir.get_tmp_directory(0),
                                                      parameters=parameters)
        else:
            monitoring.to_log_and_console("    unknown enhancement method: '"
                                          + str(parameters.intensity_enhancement) + "'", 2)
        #
        # no fusion to be done
        #
        if (parameters.intensity_transformation is None or parameters.intensity_transformation.lower() == 'none') \
                and parameters.outer_contour_enhancement is False:
            return enhanced_image

    #
    # there is an outer contour image
    # set contour_image
    #
    if parameters.outer_contour_enhancement is True:
        if (parameters.intensity_transformation is None or parameters.intensity_transformation.lower() == 'none') \
                and (parameters.intensity_enhancement is None or parameters.intensity_enhancement.lower() == 'none'):
            experiment.working_dir.make_rec_directory()
            contour_image = reconstructed_image
            monitoring.to_log_and_console("    weird, only the outer contour is used for reconstruction", 2)
        else:
            #
            # there is an other operation
            #
            contour_image = common.add_suffix(input_image, local_suffix + "_outercontour",
                                              new_dirname=experiment.working_dir.get_tmp_directory(0),
                                              new_extension=experiment.default_image_suffix)
        contour_image_bytes = 1
        #
        # compute 'contour_image'
        #
        if (not os.path.isfile(contour_image) and (reconstructed_image is None
                                                   or not os.path.isfile(reconstructed_image))) \
                or monitoring.forceResultsToBeBuilt is True:
            astec_name = experiment.astec_dir.get_image_name(current_time)
            deformed_segmentation = common.add_suffix(astec_name, '_deformed_segmentation_from_previous',
                                                      new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                      new_extension=experiment.default_image_suffix)
            previous_segmentation = experiment.get_segmentation_image(previous_time)
            if not os.path.isfile(deformed_segmentation) or monitoring.forceResultsToBeBuilt is True:
                deformation = get_deformation_from_current_to_previous(current_time, experiment, parameters,
                                                                       previous_time)
                cpp_wrapping.apply_transformation(previous_segmentation, deformed_segmentation, deformation,
                                                  interpolation_mode='nearest', monitoring=monitoring)
            cpp_wrapping.outer_contour(deformed_segmentation, contour_image, monitoring=monitoring)
        #
        # no fusion to be done
        #
        if (parameters.intensity_transformation is None or parameters.intensity_transformation.lower() == 'none') \
                and (parameters.intensity_enhancement is None or parameters.intensity_enhancement.lower() == 'none'):
            return contour_image

    #
    # here there are at least two images to be fused
    #

    experiment.working_dir.make_rec_directory()

    if parameters.reconstruction_images_combination.lower() == "maximum" \
            or parameters.reconstruction_images_combination.lower() == "max":
        arit_options = " -max"
    elif parameters.reconstruction_images_combination.lower() == "addition" \
            or parameters.reconstruction_images_combination.lower() == "add":
        arit_options = " -add -o 2"

    else:
        monitoring.to_log_and_console("    unknown image combination operation: '"
                                      + str(parameters.reconstruction_images_combination) + "'", 2)
        return None

    if intensity_image is not None:
        if enhanced_image is not None:
            if parameters.reconstruction_images_combination.lower() == "maximum" \
                    or parameters.reconstruction_images_combination.lower() == "max":
                if intensity_image_bytes == 1 and enhanced_image_bytes == 1:
                    arit_options += " -o 1"
                else:
                    arit_options += " -o 2"
            cpp_wrapping.arithmetic_operation(intensity_image, enhanced_image, reconstructed_image,
                                              other_options=arit_options, monitoring=monitoring)
            if contour_image is not None:
                if parameters.reconstruction_images_combination.lower() == "maximum" \
                        or parameters.reconstruction_images_combination.lower() == "max":
                    if intensity_image_bytes == 1 and enhanced_image_bytes == 1 and contour_image_bytes == 1:
                        arit_options += " -o 1"
                    else:
                        arit_options += " -o 2"
                cpp_wrapping.arithmetic_operation(contour_image, reconstructed_image, reconstructed_image,
                                                  other_options=arit_options, monitoring=monitoring)
            return reconstructed_image
        else:
            if contour_image is not None:
                if parameters.reconstruction_images_combination.lower() == "maximum" \
                        or parameters.reconstruction_images_combination.lower() == "max":
                    if intensity_image_bytes == 1 and contour_image_bytes == 1:
                        arit_options += " -o 1"
                    else:
                        arit_options += " -o 2"
                cpp_wrapping.arithmetic_operation(intensity_image, contour_image, reconstructed_image,
                                                  other_options=arit_options, monitoring=monitoring)
                return reconstructed_image
            else:
                monitoring.to_log_and_console("    weird. There should be a fusion to be done. Returns intensity", 2)
                return intensity_image
    elif enhanced_image is not None:
        if contour_image is not None:
            if parameters.reconstruction_images_combination.lower() == "maximum" \
                    or parameters.reconstruction_images_combination.lower() == "max":
                if enhanced_image_bytes == 1 and contour_image_bytes == 1:
                    arit_options += " -o 1"
                else:
                    arit_options += " -o 2"
            cpp_wrapping.arithmetic_operation(enhanced_image, contour_image, reconstructed_image,
                                              other_options=arit_options, monitoring=monitoring)
            return reconstructed_image
        else:
            monitoring.to_log_and_console("    weird. There should be a fusion to be done. Returns enhanced", 2)
            return enhanced_image
    elif contour_image is not None:
        monitoring.to_log_and_console("    weird. There should be a fusion to be done. Returns contours", 2)
        return contour_image

    #
    # should not reach this point
    #
    return None
