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


import ASTEC.commonTools as commonTools
import ASTEC.EMBRYOPROPERTIES as embryoProp
import ASTEC.nomenclature as nomenclature

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


#    my_parser.add_argument('-e', '--embryo-rep',
#                           action='store', dest='embryo_path', const=None,
#                           help='path to the embryo data')

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
    monitoring = commonTools.Monitoring()
    diagnosis = embryoProp.DiagnosisParameters()

    #
    # reading command line arguments
    #

    parser = ArgumentParser(description='X-lineage')
    _set_options(parser)
    args = parser.parse_args()

    monitoring.update_from_args(args)
    diagnosis.update_from_args(args)

    path_log_file = os.path.join(nomenclature.FLAG_EXECUTABLE + '-' + nomenclature.FLAG_TIMESTAMP + '.log')
    path_log_file = nomenclature.replaceTIMESTAMP(path_log_file, start_time)
    path_log_file = nomenclature.replaceEXECUTABLE(path_log_file, __file__)

    #
    # uncomment the following line to have a log file written
    #
    # monitoring.logfile = path_log_file
    embryoProp.monitoring.copy(monitoring)

    #
    # read input file(s)
    # 1. input file(s): it is assumed that there are keys describing for each dictionary entry
    # 2. lineage file: such a key may be missing
    #

    inputdict = embryoProp.read_dictionary(args.inputFiles)

    if args.print_input_types is True:
        embryoProp.print_type(inputdict, desc="root")

    if inputdict == {}:
        print "error: empty input dictionary"
        sys.exit(-1)


    #
    # display content
    #

    if args.print_content is True:
        embryoProp.print_keys(inputdict)

    #
    # is a diagnosis to be done?
    #

    if args.print_diagnosis is True:
        embryoProp.diagnosis(inputdict, args.outputFeatures, diagnosis)


    #
    # is there some comparison to be done?
    #

    if args.compareFiles is not None and len(args.compareFiles) > 0:
        comparedict = embryoProp.read_dictionary(args.compareFiles)
        if comparedict == {}:
            print "error: empty dictionary to be compared with"
        else:
            embryoProp.comparison(inputdict, comparedict, args.outputFeatures, 'input entry', 'compared entry')

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
            target_key = embryoProp.keydictionary[feature]

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
            if ofile.endswith("pkl") is True:
                propertiesfile = open(ofile, 'w')
                pkl.dump(outputdict, propertiesfile)
                propertiesfile.close()
            elif ofile.endswith("xml") is True:
                xmltree = embryoProp.dict2xml(outputdict)
                xmltree.write(ofile)
            elif ofile.endswith("tlp") is True:
                embryoProp.write_tlp_file(outputdict, ofile)
            else:
                print "   error: extension not recognized for '" + str(ofile) + "'"

    endtime = time.localtime()

    monitoring.to_log_and_console("")
    monitoring.to_log_and_console("Total execution time = "+str(time.mktime(endtime)-time.mktime(start_time))+"sec")
    monitoring.to_log_and_console("")


