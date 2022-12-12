'''
 * @author Rroscha
 * @date 2022/12/12 21:40
'''

import argparse
import csv
from multiprocessing import Pool as ThreadPool
from functools import partial
import os
import pickle
import copy
import time

import networkx as nx
from androguard.misc import AnalyzeAPK
from androguard.core.analysis.analysis import MethodAnalysis
import dill


class FeatureExtraction:
    malware_cg_dir = None
    benign_cg_dir = None
    compression_algorithm = None

    apk_dir = None
    output_dir = None

    MALWARE = 'malware'
    BENIGN = 'benign'

    SEP = '/'
    THREAD_NUMBER = 1  # 14

    SENSITIVE_APIS = './Res/mapping_5.1.1.pkl'
    APIGRAPH = './Res/method_entity_embedding_TransE.pkl'
    APIGRAPH_EMBEDDING = None

    DUMMY_MAIN_CLASS = '<dummyMainClass:'

    smali_basic_type_dict = {
        'void': 'V',
        'boolean': 'Z',
        'byte': 'B',
        'short': 'S',
        'char': 'C',
        'int': 'I',
        'long': 'J',
        'float': 'F',
        'double': 'D'
    }

    smali_basic_type_dict_reverse = {
        'V': 'void',
        'Z': 'boolean',
        'B': 'byte',
        'S': 'short',
        'C': 'char',
        'I': 'int',
        'J': 'long',
        'F': 'float',
        'D': 'double'
    }

    API_CANDIDATES = ["<android.", "<com.android.internal.util", "<dalvik.", "<java.", "<javax.", "<org.apache.",
                      "<org.json.", "<org.w3c.dom.", "<org.xml.sax", "<org.xmlpull.v1.", "<junit."]

    @staticmethod
    def set_attribute(apk_dir, malware_cg_dir, benign_cg_dir, output_dir):
        FeatureExtraction.apk_dir = apk_dir
        FeatureExtraction.malware_cg_dir = malware_cg_dir
        FeatureExtraction.benign_cg_dir = benign_cg_dir
        FeatureExtraction.output_dir = output_dir

    @staticmethod
    def get_short_name(file_path):
        return os.path.splitext(os.path.split(file_path)[1])[0]

    @staticmethod
    def transfer_smali_type_to_jimple_type(var_type):
        return_value = None

        # Resolve [
        lb_counter = str(var_type).count('[')
        first = lb_counter
        if var_type[first] == 'L':
            # Object
            return_value = var_type[first:].replace('L', '', 1).replace('/', '.').replace(';', '')  # Ljava/lang/String;
        else:
            # Basic type
            return_value = FeatureExtraction.smali_basic_type_dict_reverse[var_type[first]]

        if lb_counter > 0:
            # Array
            for i in range(lb_counter):
                return_value += '[]'

        return str(return_value)

    @staticmethod
    def transfer_smali_method_to_jimple_method(method_: str):

        return_value = ''

        first = 0
        last = method_.index(' ')

        # class
        class_name = method_[first + 1:last - 1].replace('/', '.')

        # method name
        first = last + 1
        last = method_.index(' ', first)
        method_name = method_[first:last]

        # params
        first = last + 2
        last = method_.index(')', first)
        params = method_[first:last]
        params = params.split(' ')
        params_ = []

        for p in params:
            if p != '':
                params_.append(FeatureExtraction.transfer_smali_type_to_jimple_type(p))

        # return type
        first = last + 1
        return_type = method_[first:]
        return_type = FeatureExtraction.transfer_smali_type_to_jimple_type(return_type)

        # Assemble
        return_value += '<'
        return_value += class_name
        return_value += ': '
        return_value += return_type
        return_value += ' '
        return_value += method_name
        return_value += '('

        max_index = len(params_) - 1
        i = 0
        for p in params_:
            if i == max_index:
                return_value += p
            else:
                return_value += p
                return_value += ','
            i += 1

        return_value += ')>'

        return return_value

    @staticmethod
    def starts_with_api(node):
        is_api = False

        for candidate in FeatureExtraction.API_CANDIDATES:
            if node.startswith(candidate):
                is_api = True
                break

        return is_api

    @staticmethod
    def complement_api(primitive_graph, apk_path):
        apk, dalvik_vm_format, analysis = AnalyzeAPK(apk_path)

        # Get all api methods (androguard)
        apis_set = set()

        for method in analysis.get_methods():
            if MethodAnalysis.is_android_api(method):  # method.is_android_api()
                # print(method.method.full_name)
                apis_set.add(method)

        apis_androguard = list(apis_set)

        # Transfer apis to another format
        temp = []
        for a in apis_androguard:
            temp.append(FeatureExtraction.transfer_smali_method_to_jimple_method(str(a.full_name)))

        apis_androguard = temp

        # Get all api methods in graph
        nodes_list_with_apis = list(primitive_graph)
        api_list_in_graph = []

        for n in nodes_list_with_apis:
            is_api = FeatureExtraction.starts_with_api(n)
            primitive_graph.nodes[n]["API"] = is_api

            if is_api:
                api_list_in_graph.append(n)

        api_list_in_graph = list(set(api_list_in_graph))

        # Read api use permission
        with open(FeatureExtraction.SENSITIVE_APIS, 'rb') as f:
            sensitive_apis = pickle.load(f)

        fail_analyzed_apis = list(set(apis_androguard) - set(api_list_in_graph))

        call_graph_temp = analysis.get_call_graph()
        nx_graph_temp = copy.deepcopy(primitive_graph)

        g = nx.MultiDiGraph()
        waiting_append = []
        for x in list(call_graph_temp.edges):
            t = list(x)
            t[0] = FeatureExtraction.transfer_smali_method_to_jimple_method(str(t[0].full_name))
            t[1] = FeatureExtraction.transfer_smali_method_to_jimple_method(str(t[1].full_name))
            waiting_append.append(tuple(t))

        g.add_edges_from(waiting_append)

        reserved_nodes = []
        for x in fail_analyzed_apis:
            reserved_nodes.extend(list(g.predecessors(x)))

        reserved_nodes_temp = list(set(reserved_nodes))
        reserved_nodes = []

        nx_graph_nodes = list(primitive_graph.nodes)
        for x in reserved_nodes_temp:
            if x in nx_graph_nodes:
                reserved_nodes.append(x)

        g_s = g.subgraph(reserved_nodes + fail_analyzed_apis)

        g_s_nodes = list(g_s.nodes)
        g_s_removed_nodes = []
        for n in g_s_nodes:
            if g_s.in_degree(n) + g_s.out_degree(n) == 0:
                g_s_removed_nodes.append(n)
        g_s = nx.MultiDiGraph(g_s)
        g_s.remove_nodes_from(g_s_removed_nodes)

        supplemented_graph = nx.compose(nx_graph_temp, g_s)

        final_nodes = list(supplemented_graph)
        final_api_nodes = [x for x in final_nodes if FeatureExtraction.starts_with_api(x)]

        for n in final_api_nodes:
            second_space_index = n.index(' ', n.index(' ') + 1)
            first = second_space_index + 1
            last = n.index('(')
            api = n[1:n.index(':')] + '.' + n[first:last]

            supplemented_graph.nodes[n]["API"] = True

            if api in sensitive_apis:
                supplemented_graph.nodes[n]['is_sensitive'] = True
            else:
                supplemented_graph.nodes[n]['is_sensitive'] = False

        return supplemented_graph

    @staticmethod
    def transfer_jimple_method2full_format(jimple_method):
        second_space_index = jimple_method.index(' ', jimple_method.index(' ') + 1)
        first = second_space_index + 1
        last = jimple_method.index('(')
        full_format = jimple_method[1:jimple_method.index(':')] + '.' + jimple_method[first:last]

        return full_format

    @staticmethod
    def run_internal(file_path, label, output_dir, semi_apk_path):
        start_time = time.time()

        SEP = '/'

        # Read graph
        try:
            primitive_graph = nx.nx_pydot.read_dot(file_path)
        except Exception as e:
            print(e)
            return

        # Reconstruct graph (Complement APIs)
        supplemented_graph = FeatureExtraction.complement_api(primitive_graph,
                                                        semi_apk_path + SEP +
                                                        FeatureExtraction.get_short_name(file_path)[:-10] + '.apk')

        print('Working on %s.' % FeatureExtraction.get_short_name(file_path)[:-10])

        # Compression
        # Find sensitive apis
        apk_sensitive_apis = []
        for n in list(supplemented_graph.nodes):
            if supplemented_graph.nodes[n]['API'] and supplemented_graph.nodes[n]['is_sensitive']:
                apk_sensitive_apis.append(n)

        # Compute centrality
        # Degree Closeness Katz Harmonic
        # Degree
        print('    Compute degree...')
        degree_centrality_dict = nx.degree_centrality(supplemented_graph)

        for n in degree_centrality_dict.keys():
            supplemented_graph.nodes[n]['degree'] = degree_centrality_dict[n]

        # Closeness
        print('    Compute closeness...')
        closeness_centrality_dict = nx.closeness_centrality(supplemented_graph)

        for n in closeness_centrality_dict.keys():
            supplemented_graph.nodes[n]['closeness'] = closeness_centrality_dict[n]

        # Katz
        print('    Compute katz...')
        katz_centrality_dict = nx.katz_centrality(nx.DiGraph(supplemented_graph))

        for n in katz_centrality_dict.keys():
            supplemented_graph.nodes[n]['katz'] = closeness_centrality_dict[n]

        # Harmonic
        print('    Compute harmonic...')
        harmonic_centrality_dict = nx.harmonic_centrality(supplemented_graph)

        for n in harmonic_centrality_dict.keys():
            supplemented_graph.nodes[n]['harmonic'] = harmonic_centrality_dict[n]

        # Save
        apk_name = FeatureExtraction.get_short_name(file_path)[:-10]
        output_path = output_dir + SEP + apk_name + '.pkl'
        with open(output_path, 'wb') as f:
            dill.dump([label, supplemented_graph], f)

    @staticmethod
    def run():
        # Get all file paths of cgs
        malware_cg_path_list = None
        benign_cg_path_list = None

        # Malware
        fileList = os.listdir(FeatureExtraction.malware_cg_dir)
        malware_cg_path_list = [FeatureExtraction.malware_cg_dir + FeatureExtraction.SEP + f for f in fileList]

        # Benign
        fileList = os.listdir(FeatureExtraction.benign_cg_dir)
        benign_cg_path_list = [FeatureExtraction.benign_cg_dir + FeatureExtraction.SEP + f for f in fileList]

        # Generate thread pools
        malicious_pools = ThreadPool(FeatureExtraction.THREAD_NUMBER)
        benign_pools = ThreadPool(FeatureExtraction.THREAD_NUMBER)

        print("---- Compress malware ----")
        malicious_pools.map(partial(FeatureExtraction.run_internal,
                                    label=1,
                                    output_dir=FeatureExtraction.output_dir + FeatureExtraction.SEP +
                                               FeatureExtraction.MALWARE,
                                    semi_apk_path=FeatureExtraction.apk_dir + FeatureExtraction.SEP + 
                                                  FeatureExtraction.MALWARE),
                            malware_cg_path_list)

        print("---- Compress benign ----")
        benign_pools.map(partial(FeatureExtraction.run_internal,
                                 label=0,
                                 output_dir=FeatureExtraction.output_dir + FeatureExtraction.SEP + 
                                            FeatureExtraction.BENIGN,
                                 semi_apk_path=FeatureExtraction.apk_dir + FeatureExtraction.SEP + 
                                               FeatureExtraction.BENIGN),
                         benign_cg_path_list)

        print("All finished.")

    @staticmethod
    def read_callgraph(file):
        CG = nx.read_gexf(file)
        return CG

    @staticmethod
    def obtain_sensitive_apis(file):
        sensitive_apis = []
        with open(file, 'r') as f:
            for line in f.readlines():
                if line.strip() == '':
                    continue
                else:
                    sensitive_apis.append(line.strip())
        return sensitive_apis

    @staticmethod
    def katz_centrality_feature(file, sensitive_apis):
        app_name = file.split('/')[-1][:-5]
        print(app_name)

        vector = []

        try:
            CG = FeatureExtraction.read_callgraph(file)
            node_centrality = nx.katz_centrality(nx.DiGraph(CG))

            for api in sensitive_apis:
                if api in node_centrality.keys():
                    vector.append(node_centrality[api])
                else:
                    vector.append(0)
        except Exception as e:
            for api in sensitive_apis:
                vector.append(0)

        return (app_name, vector)

    @staticmethod
    def run_an_internal(centrality, sensitive_apis):
        Vectors  = []
        Labels   = []

        apps_b = [FeatureExtraction.benign_cg_dir + '/' + i for i in os.listdir(FeatureExtraction.benign_cg_dir)]
        apps_m = [FeatureExtraction.malware_cg_dir + '/' + i for i in os.listdir(FeatureExtraction.malware_cg_dir)]


        pool_b   = ThreadPool(14)
        pool_m   = ThreadPool(14)

        vector_b = pool_b.map(partial(FeatureExtraction.katz_centrality_feature, sensitive_apis=sensitive_apis), apps_b)
        vector_m = pool_m.map(partial(FeatureExtraction.katz_centrality_feature, sensitive_apis=sensitive_apis), apps_m)

        Vectors.extend(vector_b)
        Labels.extend([0 for i in range(len(vector_b))])

        Vectors.extend(vector_m)
        Labels.extend([1 for i in range(len(vector_m))])

        return Vectors, Labels

    @staticmethod
    def run_an():
        sensitive_apis_path = 'sensitive_apis.txt'
        sensitive_apis = FeatureExtraction.obtain_sensitive_apis(sensitive_apis_path)

        centrality = 'katz'

        Vectors, Labels = FeatureExtraction.run_an_internal(centrality, sensitive_apis)

        feature_csv = [[] for i in range(len(Labels) + 1)]
        feature_csv[0].append('app')
        feature_csv[0].extend(sensitive_apis)
        feature_csv[0].append('Label')

        for i in range(len(Labels)):
            (app, vector) = Vectors[i]
            feature_csv[i + 1].append(app)
            feature_csv[i + 1].extend(vector)
            feature_csv[i + 1].append(Labels[i])

        if not os.path.exists(FeatureExtraction.output_dir):
            os.makedirs(FeatureExtraction.output_dir)

        csv_path = FeatureExtraction.output_dir + '/' + centrality + '_features.csv'

        with open(csv_path, 'w', newline='') as f:
            csvfile = csv.writer(f)
            csvfile.writerows(feature_csv)

if __name__ == '__main__':
    CALL_GRAPH_MODE = 2

    import argparse

    import argparse

    parser = argparse.ArgumentParser(description='Compress graph.')
    parser.add_argument('-a',
                        help='APK directory.')
    parser.add_argument('-m',
                        help='Malware cg directory.')
    parser.add_argument('-b',
                        help='Benign cg directory.')
    parser.add_argument('-o',
                        help='Directory of outputting features.')

    args = parser.parse_args()

    FeatureExtraction.set_attribute(args.a, args.m, args.b, args.o)

    if CALL_GRAPH_MODE == 1:
        FeatureExtraction.run()
    elif CALL_GRAPH_MODE == 2:
        FeatureExtraction.run_an()



