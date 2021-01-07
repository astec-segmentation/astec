#!/usr/bin/python2.7

import os
import cPickle as pkl
import time
import sys

from argparse import ArgumentParser

#
# local imports
# add ASTEC subdirectory
#


import ASTEC.common as common
import ASTEC.properties as properties
from ASTEC.CommunFunctions.cpp_wrapping import path_to_vt
#
#
#
#
#


def _set_options(my_parser):
    proc = "_set_options"
    if not isinstance(my_parser, ArgumentParser):
        print proc + ": argument is not of type ArgumentParser"
        return
    #
    # common parameters
    #

    my_parser.add_argument('-p', '--parameters',
                           action='store', dest='parameterFile', const=None,
                           help='python file containing parameters definition')
    my_parser.add_argument('-e', '--embryo-rep',
                           action='store', dest='embryo_path', const=None,
                           help='path to the embryo data')

    #
    # other options
    #

    my_parser.add_argument('-i', '--input',
                           action='store', nargs='*', dest='inputFiles', const=None,
                           help='pkl or xml file(s)')

    my_parser.add_argument('-o', '--output',
                           action='store', nargs='*', dest='outputFiles', const=None,
                           help='pkl file containing the lineage')

    my_parser.add_argument('-c', '--compare',
                           action='store', nargs='*', dest='compareFiles', const=None,
                           help='pkl or xml file(s), to be compared to those of "--input"')

    my_parser.add_argument('-feature', '-property',
                           action='store', nargs='*', dest='outputFeatures', const=None,
                           help="features to be extracted from the lineage: 'lineage', 'h_min', 'volume', 'surface'" +
                                ", 'sigma', 'label_in_time', 'barycenter', 'fate', 'fate2', 'fate3', 'fate4'" +
                                ", 'all-cells', 'principal-value', 'name', 'contact', 'history', 'principal-vector'" +
                                ", 'name-score', 'cell-compactness'")

    my_parser.add_argument('--check', '--check-volume-lineage',
                           action='store_const', dest='check_volume_lineage',
                           default=False, const=True,
                           help='perform some tests')

    my_parser.add_argument('--diagnosis',
                           action='store_const', dest='print_diagnosis',
                           default=False, const=True,
                           help='perform some tests')

    my_parser.add_argument('--diagnosis-minimal-volume',
                           action='store', dest='diagnosis_minimal_volume',
                           help='displays all cells with smaller volume')

    my_parser.add_argument('--diagnosis-items',
                           action='store', dest='diagnosis_items',
                           help='minimal number of items to be displayed')

    my_parser.add_argument('--print-content', '--print-keys',
                           action='store_const', dest='print_content',
                           default=False, const=True,
                           help='print keys of the input file(s) (read as dictionary)')

    my_parser.add_argument('--print-types',
                           action='store_const', dest='print_input_types',
                           default=False, const=True,
                           help='print types of read features (for debug purpose)')

    my_parser.add_argument('--old-keys',
                           action='store_const', dest='old_keys',
                           default=False, const=True,
                           help='use historical keys for pkl file')

    #
    # control parameters
    #

    my_parser.add_argument('-k', '--keep-temporary-files',
                           action='store_const', dest='keepTemporaryFiles',
                           default=False, const=True,
                           help='keep temporary files')

    my_parser.add_argument('-f', '--force',
                           action='store_const', dest='forceResultsToBeBuilt',
                           default=False, const=True,
                           help='force building of results')

    my_parser.add_argument('-v', '--verbose',
                           action='count', dest='verbose', default=1,
                           help='incrementation of verboseness')
    my_parser.add_argument('-nv', '--no-verbose',
                           action='store_const', dest='verbose', const=0,
                           help='no verbose at all')
    my_parser.add_argument('-d', '--debug',
                           action='count', dest='debug', default=0,
                           help='incrementation of debug level')
    my_parser.add_argument('-nd', '--no-debug',
                           action='store_const', dest='debug', const=0,
                           help='no debug information')

    return


#
#
# main 
#
#


if __name__ == '__main__':

    #
    # initialization
    #

    start_time = time.localtime()
    monitoring = common.Monitoring()
    experiment = common.Experiment()

    #
    # reading command line arguments
    # and update from command line arguments
    #
    parser = ArgumentParser(description='X-embryoproperties.py')
    _set_options(parser)
    args = parser.parse_args()

    monitoring.update_from_args(args)
    experiment.update_from_args(args)

    diagnosis = properties.DiagnosisParameters()
    diagnosis.update_from_args(args)

    #
    # is there a parameter file?
    # if yes, compute properties from an image sequence
    #
    if args.parameterFile is not None and os.path.isfile(args.parameterFile):

        #
        # reading parameter files
        # and updating parameters
        #
        parameter_file = common.get_parameter_file(args.parameterFile)
        experiment.update_from_parameter_file(parameter_file)

        #
        # set
        # 1. the working directory
        #    that's where the logfile will be written
        # 2. the log file name
        #    it creates the logfile dir, if necessary
        #
        experiment.working_dir = experiment.intrareg_dir
        monitoring.set_log_filename(experiment, __file__, start_time)

        #
        # keep history of command line executions
        # and copy parameter file
        #
        experiment.update_history_at_start(__file__, start_time, parameter_file, path_to_vt())
        experiment.copy_stamped_file(start_time, parameter_file)

        #
        # copy monitoring information into other "files"
        # so the log
        #
        # filename is known
        #
        common.monitoring.copy(monitoring)

        #
        # write generic information into the log file
        #
        monitoring.write_configuration()
        experiment.write_parameters()

        ############################################################
        #
        # specific part
        #
        ############################################################

        #
        # copy monitoring information into other "files"
        # so the log filename is known
        #
        properties.monitoring.copy(monitoring)

        #
        # manage parameters
        # 1. initialize
        # 2. update parameters
        # 3. write parameters into the logfile
        #

        parameters = properties.CellPropertiesParameters()
        parameters.update_from_parameter_file(parameter_file)

        parameters.write_parameters(monitoring.log_filename)
        diagnosis.write_parameters(monitoring.log_filename)

        #
        # compute sequence properties in xml format
        #
        xml_output = properties.property_computation(experiment)
        if xml_output is None:
            monitoring.to_log_and_console('    error during properties computation')
            sys.exit(-1)

        #
        # prepare the copy of the sequence properties in pkl format
        #
        pkl_output = xml_output[0:len(xml_output)-4] + ".pkl"
        pkl_is_to_be_done = True

        if os.path.isfile(pkl_output):
            if not monitoring.forceResultsToBeBuilt:
                monitoring.to_log_and_console('    pkl file already existing', 2)
                pkl_is_to_be_done = False
            else:
                monitoring.to_log_and_console('    pkl file already existing, but forced', 2)

        #
        # prepare the copy of the sequence properties in tlp format
        #
        tlp_output = xml_output[0:len(xml_output)-4] + ".tlp"
        tlp_is_to_be_done = True

        if os.path.isfile(tlp_output):
            if not monitoring.forceResultsToBeBuilt:
                monitoring.to_log_and_console('    tlp file already existing', 2)
                tlp_is_to_be_done = False
            else:
                monitoring.to_log_and_console('    tlp file already existing, but forced', 2)

        #
        # copy properties
        #

        if pkl_is_to_be_done is True or tlp_is_to_be_done is True:
            inputdict = properties.read_dictionary(xml_output)
            if pkl_is_to_be_done is True:
                properties.write_dictionary(pkl_output, inputdict)
            if tlp_is_to_be_done is True:
                properties.write_dictionary(tlp_output, inputdict)
            del inputdict

        endtime = time.localtime()

        monitoring.to_log_and_console("")
        monitoring.to_log_and_console("Total execution time = "+str(time.mktime(endtime)-time.mktime(start_time))+"sec")
        monitoring.to_log_and_console("")

    else:

        properties.monitoring.copy(monitoring)

        #
        # read input file(s)
        # 1. input file(s): it is assumed that there are keys describing for each dictionary entry
        # 2. lineage file: such a key may be missing
        #

        inputdict = properties.read_dictionary(args.inputFiles, inputpropertiesdict={})

        if args.print_input_types is True:
            properties.print_type(inputdict, desc="root")

        if inputdict == {}:
            monitoring.to_log_and_console("error: empty input dictionary")
            sys.exit(-1)

        #
        # display content
        #

        if args.print_content is True:
            properties.print_keys(inputdict, desc="input dictionary")

        #
        # is a diagnosis to be done?
        #

        if args.print_diagnosis is True:
            properties.diagnosis(inputdict, args.outputFeatures, diagnosis)


        # is a check to be done?
        #

        if args.check_volume_lineage is True:
            properties.check_volume_lineage(inputdict)

        #
        # is there some comparison to be done?
        #
        if args.compareFiles is not None and len(args.compareFiles) > 0:
            comparedict = properties.read_dictionary(args.compareFiles, inputpropertiesdict={})
            if args.print_content is True:
                properties.print_keys(comparedict, desc="dictionary to be compared with")
            if comparedict == {}:
                print "error: empty dictionary to be compared with"
            else:
                properties.comparison(inputdict, comparedict, args.outputFeatures, 'input entry', 'compared entry')

        #
        # select features if required
        #

        outputdict = {}

        if args.outputFeatures is not None:

            #
            # search for required features
            #

            for feature in args.outputFeatures:

                # print "search feature '" + str(feature) + "'"
                target_key = properties.keydictionary[feature]

                for searchedkey in target_key['input_keys']:
                    if searchedkey in inputdict:
                        # print "found feature '" + str(ok) + "'"
                        outputdict[target_key['output_key']] = inputdict[searchedkey]
                        break
                else:
                    print "error: feature '" + str(feature) + "' not found in dictionary"

        else:

            #
            # copy dictionary
            #

            # print "copy dictionary"
            if args.old_keys is True:
                outputdict = {}
                for inputkey in inputdict:
                    foundkey = False
                    for k in properties.keydictionary:
                        if inputkey in properties.keydictionary[k]['input_keys']:
                            outputkey = properties.keydictionary[k]['output_key']
                            if outputkey == 'cell_lineage':
                                outputdict['lin_tree'] = inputdict[inputkey]
                            elif outputkey == 'cell_volume':
                                outputdict['volumes_information'] = inputdict[inputkey]
                            else:
                                outputdict[outputkey] = inputdict[inputkey]
                            foundkey = True
                            break
                    if foundkey is False:
                        outputdict[inputkey] = inputdict[inputkey]
            else:
                outputdict = inputdict

        if outputdict == {}:
            print "error: empty input dictionary ?! ... exiting"
            sys.exit()

        #
        # produces outputs
        #

        if args.outputFiles is None:
            pass
            # print "error: no output file(s)"
        else:
            for ofile in args.outputFiles:
                print "... writing '" + str(ofile) + "'"
                properties.write_dictionary(ofile, outputdict)

        endtime = time.localtime()

        # monitoring.to_log_and_console("")
        # monitoring.to_log_and_console("Total execution time = "+str(time.mktime(endtime)-time.mktime(start_time))+"sec")
        # monitoring.to_log_and_console("")
