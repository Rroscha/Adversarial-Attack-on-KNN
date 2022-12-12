'''
 * @author Rroscha
 * @date 2022/12/12 21:27
'''

import sys
sys.path.append(r'E:\\T-WorkSpace\\AaKnn\\AMD')

import logging
import os
import random
import re
import shelve
import pickle
import shutil
from multiprocessing import Pool as ThreadPool
from functools import partial

import networkx as nx
from gensim.models.word2vec import Word2Vec, PathLineSentences
from gensim.models.word2vec import LineSentence
import pydot
import networkx as nx
from androguard.misc import AnalyzeAPK
from androguard.core.analysis.analysis import MethodAnalysis

from Util.CallGraphGenerator import CallGraphGenerator

class DataPrepare:
    apk_directory_path = None
    apks_number = None
    fcg_dot_output_path = None
    files_paths = None

    ic3_output = None
    dbname = "dialdroid"

    apk_extend = '.apk'
    dot_extend = '.dot'
    flowdroid_extend = '_flowdroid'

    semi_platform_path = './android-platforms-master'

    callgraph_flowdroid_path = './Bin/CallGraphGenerator.jar'

    SEP = '/'
    PKL = ".pkl"

    xms = "2G"
    xmx = "4G"
    ic3_Dialdroid = "./Bin/ICC/ic3-dialdroid.jar"




    @staticmethod
    def set_attribute(apk_d, cg, ic3_o):
        DataPrepare.apk_directory_path = apk_d
        DataPrepare.fcg_dot_output_path = cg

        DataPrepare.ic3_output = ic3_o

    @staticmethod
    def get_all_files_paths():
        fileList = os.listdir(DataPrepare.apk_directory_path)
        DataPrepare.files_paths = [DataPrepare.apk_directory_path + DataPrepare.SEP + f for f in fileList]

    @staticmethod
    def get_file_short_name(file_path):
        return os.path.splitext(os.path.split(file_path)[1])[0]

    @staticmethod
    def generate_fcg(file_path, counter_fcg):
        print("Generate function call graph: " + file_path + " (" + str(counter_fcg) + ")")

        # Initialize object ApkStcInfo
        apk_info = CallGraphGenerator()
        apk_info.initialize_androguard(apk_path=file_path)

        TARGET_JAR_1 = str(apk_info.get_apk_targetsdkversion_androguard())
        if TARGET_JAR_1 == "20":
            TARGET_JAR_1 = "21"

        CallGraphGenerator.set_callgraph_flowdroid_jar_path(DataPrepare.callgraph_flowdroid_path)
        CallGraphGenerator.set_callgraph_jar_args(file_path,
                                                  DataPrepare.semi_platform_path,
                                                  TARGET_JAR_1,
                                                  DataPrepare.fcg_dot_output_path)

        output_dot_path_flowdroid = DataPrepare.fcg_dot_output_path + DataPrepare.SEP + DataPrepare.get_file_short_name(
            file_path) + DataPrepare.flowdroid_extend + DataPrepare.dot_extend

        if not os.path.exists(output_dot_path_flowdroid):
            apk_short_name = DataPrepare.get_file_short_name(file_path)

            # ic3 first
            android_Jar = DataPrepare.semi_platform_path + DataPrepare.SEP + "android-" + \
                          TARGET_JAR_1 + DataPrepare.SEP + "android.jar"

            def CHECK_IC3_RESULT_PATH():
                semi_ic3_path = DataPrepare.ic3_output + DataPrepare.SEP + apk_short_name
                ic3_result_path = None
                is_timeout = False

                if not os.path.exists(semi_ic3_path):
                    os.makedirs(semi_ic3_path)

                for file in os.listdir(semi_ic3_path):
                    if file == "timeout.txt":
                        is_timeout = True
                        break
                    if os.path.splitext(file)[1] == ".dat":
                        ic3_result_path = semi_ic3_path + DataPrepare.SEP + file
                        break

                if is_timeout:
                    ic3_result_path = "TimeOut"
                    is_success = False
                    return ic3_result_path, is_success

                is_success = None
                if ic3_result_path is None:
                    is_success = False
                else:
                    is_success = True
                return ic3_result_path, is_success

            ic3_data_path, success = CHECK_IC3_RESULT_PATH()

            if not success and not ic3_data_path == "TimeOut":
                print("Generating ICC for " + apk_short_name + ".apk!")
                isBinary = True
                ic3_data_path, success = apk_info.get_apk_intentcominfo_ic3_dialdroid(DataPrepare.xms, DataPrepare.xmx,
                                                                                      DataPrepare.ic3_Dialdroid,
                                                                                      file_path, android_Jar,
                                                                                      DataPrepare.dbname,
                                                                                      DataPrepare.ic3_output, isBinary)
            icc_model = "NULL"

            if success:
                print("ICC Success!")
                icc_model = ic3_data_path
            else:
                if ic3_data_path == "TimeOut":
                    timeout_path = DataPrepare.ic3_output + DataPrepare.SEP + apk_short_name + DataPrepare.SEP + \
                                   "timeout.txt"
                    tf = open(timeout_path, "w")
                    tf.close()
                logging.warning("Failed to generate ICC for " + apk_short_name + ".apk!")
                # print("Failed to generate ICC for " + apk_short_name + ".apk!")

            CallGraphGenerator.set_icc_model(icc_model)

            CallGraphGenerator.exec_callgraph_flowdroid_jar()

        if os.path.exists(output_dot_path_flowdroid):
            logging.warning("Call Graph of " + file_path + " has been generated!")

    @staticmethod
    def generate_all_fcgs():
        counter_fcg = 1
        for fp in DataPrepare.files_paths:
            try:
                DataPrepare.generate_fcg(fp, counter_fcg)
                counter_fcg += 1
            except Exception as e:
                print(e)

    @staticmethod
    def run():
        # Get files paths in files_paths
        DataPrepare.get_all_files_paths()  # files_paths

        # Generating all FCGs   ${WorkPath}/sootOutput/xxxxxxxxx.dot
        print("-------------------------CG-------------------------")
        DataPrepare.generate_all_fcgs()
        print("****************************************************")

    @staticmethod
    def extract_call_graphs(app_path, out_path):
        apk_name = app_path.split('/')[-1].split('.apk')[0]

        try:
            print(apk_name)

            apk, dalvik_vm_format, analysis = AnalyzeAPK(app_path)
            cg = analysis.get_call_graph()

            output_cg = out_path + '/' + apk_name + '.gexf'
            nx.write_gexf(cg, output_cg)
        except Exception as e:
            print(e)
            return

    @staticmethod
    def run_an():
        # Get files paths in files_paths
        DataPrepare.get_all_files_paths()  # files_paths

        apk_file_paths = DataPrepare.files_paths

        pool = ThreadPool(12)
        pool.map(partial(DataPrepare.extract_call_graphs, out_path=DataPrepare.fcg_dot_output_path),
                 apk_file_paths)


# Generate Call Graphs
if __name__ == '__main__':
    CALL_GRAPH_MODE = 2

    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--directory',
                        help='Apk directory.')
    parser.add_argument('--cg',
                        help='CG output dir.')
    parser.add_argument('--temporary',
                        help='IC3 temporary output dir.')

    args = parser.parse_args()

    if not args.directory or not args.temporary or not args.cg:
        logging.error("Missing directory or temporary path.")
        exit(1)

    if CALL_GRAPH_MODE == 1:
        DataPrepare.set_attribute(args.directory, args.cg, args.temporary)
        DataPrepare.run()
    elif CALL_GRAPH_MODE == 2:
        # Using androguard to extract CGs
        DataPrepare.run_an()




