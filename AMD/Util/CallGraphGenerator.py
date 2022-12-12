from enum import Enum, unique
import re
import subprocess
import os
import signal
import logging

from androguard.misc import AnalyzeAPK  # androguard package
# https://androguard.readthedocs.io/en/latest/api/androguard.html
from androguard.core.analysis import analysis  # Quick positioning
from androguard.core.analysis.analysis import MethodAnalysis
import pydot

class CallGraphGenerator:
    timeout = 150

    win_cmd = "taskkill /F /IM java.exe"
    OS = "Windows"  # Linux

    android_ui_class_rex = (
        'Landroid/view/.*;',  # Common
        'Landroid/widget/.*;',
        'Landroid/support/.*/app/Fragment.*;',  # support
        'Landroid/support/.*/widget/.*;',
        'Landroid/support/.*/view/.*;',
        'Landroidx/.*/app/Fragment.*;',  # androidx
        'Landroidx/.*/view/.*;',
        'Landroidx/.*/widget/.*;',
    )

    dangerous_permission_list = (
        'android.permission.ACCEPT_HANDOVER',
        'android.permission.ACCESS_BACKGROUND_LOCATION',
        'android.permission.ACCESS_COARSE_LOCATION',
        'android.permission.ACCESS_FINE_LOCATION',
        'android.permission.ACCESS_MEDIA_LOCATION',
        'android.permission.ACTIVITY_RECOGNITION',
        'com.android.voicemail.permission.ADD_VOICEMAIL',
        'android.permission.ANSWER_PHONE_CALLS',
        'android.permission.BLUETOOTH_ADVERTISE',
        'android.permission.BLUETOOTH_CONNECT',
        'android.permission.BLUETOOTH_SCAN',
        'android.permission.BODY_SENSORS',
        'android.permission.BODY_SENSORS_BACKGROUND',
        'android.permission.CALL_PHONE',
        'android.permission.CAMERA',
        'android.permission.GET_ACCOUNTS',
        'android.permission.NEARBY_WIFI_DEVICES',
        'android.permission.POST_NOTIFICATIONS',
        'android.permission.PROCESS_OUTGOING_CALLS',
        'android.permission.READ_CALENDAR',
        'android.permission.READ_CALL_LOG',
        'android.permission.READ_CONTACTS',
        'android.permission.READ_EXTERNAL_STORAGE',
        'android.permission.READ_PHONE_NUMBERS',
        'android.permission.READ_PHONE_STATE',
        'android.permission.READ_SMS',
        'android.permission.RECEIVE_MMS',
        'android.permission.RECEIVE_SMS',
        'android.permission.RECEIVE_WAP_PUSH',
        'android.permission.RECORD_AUDIO',
        'android.permission.SEND_SMS',
        'android.permission.USE_SIP',
        'android.permission.UWB_RANGING',
        'android.permission.WRITE_CALENDAR',
        'android.permission.WRITE_CALL_LOG',
        'android.permission.WRITE_CONTACTS',
        'android.permission.WRITE_EXTERNAL_STORAGE'
    )

    ic3_dialdroid_cmd = "java -Xms%s -Xmx%s -jar %s -input %s -cp %s -dbname %s -out %s %s"
    callgraph_flowdroid_cmd = 'java -jar %s %s %s %s %s %s'

    icc_model = "NULL"
    CALLGRAPH_FlowDroid_JAR_PATH = None
    semi_platform_path = None
    api_level = None
    apk_path = None
    a_output = None

    dot_data = None
    dot_node = None
    dot_nodes = None
    dot_nodes_length = None

    def __init__(self):
        self.analysis = None
        self.dalvik_vm_format = None
        self.apk = None

    def load_androguard(self, apk_path):
        assert apk_path is not None and isinstance(apk_path, str), 'No apk-file path'
        return AnalyzeAPK(apk_path)

    def initialize_androguard(self, apk_path):
        self.apk, self.dalvik_vm_format, self.analysis = self.load_androguard(apk_path)

    def get_apk_targetsdkversion_androguard(self):
        return self.apk.get_effective_target_sdk_version()

    @staticmethod
    def get_apk_intentcominfo_ic3_dialdroid(xms, xmx, ic3_dialdroid, apk_file,
                                            android_jar, dbname, sami_output_dir, is_binary):
        """
        Use ic3_dialdroid.jar to generate intent communication info in a dat file for further processing.

        :param xms: java -Xms.
        :param xmx: java -Xmx.
        :param ic3_dialdroid: ic3_dialdroid.jar, please use v2.0, which is updated by rroscha.
        :param apk_file: APK file directory.
        :param android_jar: Android.jar. Android Platform.
        :param dbname: Database Name.
        :param sami_output_dir: Half output directory. E.g. samiOutPutDir('./Output_Rroscha') + '/' + apkFileName.
        :param is_binary: Boolean.
        :return: dat/txt file's path.
        """

        if not os.path.exists(sami_output_dir):
            os.mkdir(sami_output_dir)

        # Get apk file name to construct the storing directory
        dir_name, file_name = os.path.split(apk_file)
        f_name, fe_name = os.path.splitext(file_name)

        # Use fName
        output_path = sami_output_dir + '/' + f_name  # The output path for ic3_dialdroid
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        binary = ""
        test_extent = ".txt"
        if is_binary:
            binary = "-binary"
            test_extent = ".dat"

        # Invoke ic3_dialdroid
        '''
        self.ic3_dialdroid_cmd = "java -Xms? -Xmx? -jar ? -input ? -cp ? -dbname ? -out ? ?"
        # java -Xms2G -Xmx4G -jar .\ic3-dialdroid.jar -input .\apkFile\test3.apk -cp .\android-19\android.jar -dbname dialdroid -out .\OutputTest\test3 -binary
        '''
        timeout = False

        # if self.OS == "Windows":
        #     SHELL = True
        # else:
        #     SHELL = False

        if CallGraphGenerator.OS == "Windows":
            process = subprocess.Popen(CallGraphGenerator.ic3_dialdroid_cmd % (xms, xmx, ic3_dialdroid, apk_file,
                                                                 android_jar, dbname, output_path, binary),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       stdin=subprocess.PIPE,
                                       shell=True)
        else:
            process = subprocess.Popen(CallGraphGenerator.ic3_dialdroid_cmd % (xms, xmx, ic3_dialdroid, apk_file,
                                                                 android_jar, dbname, output_path, binary),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       stdin=subprocess.PIPE,
                                       shell=True, preexec_fn=os.setsid)

        try:
            # stdout, stderr = process.communicate()
            process.communicate(timeout=CallGraphGenerator.timeout)  # timeout=ApkStcInfo.timeout
        except Exception as e:
            timeout = True
            process.kill()
            process.terminate()
            process.wait()

            if CallGraphGenerator.OS == "Windows":
                logging.warning("Windows ICC Time Out!")
                os.system(CallGraphGenerator.win_cmd)
            else:
                logging.warning("Linux ICC Time Out!")
                os.killpg(process.pid, signal.SIGTERM)
            logging.warning("ICC Generating time out!")

        if timeout:
            return "TimeOut", False

        # Check if success (Whether there is a txt/dat file in output path)
        is_success = False
        dat_path = None  # dat or txt path
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if os.path.splitext(file)[1] == test_extent:
                    is_success = True
                    dat_path = output_path + '/' + file
        return dat_path, is_success

    @staticmethod
    def set_callgraph_flowdroid_jar_path(path):
        CallGraphGenerator.CALLGRAPH_FlowDroid_JAR_PATH = path

    @staticmethod
    def set_callgraph_jar_args(a_path, s_path, a_level, a_output="./sootOutput"):
        CallGraphGenerator.apk_path = a_path
        CallGraphGenerator.semi_platform_path = s_path
        CallGraphGenerator.api_level = a_level
        CallGraphGenerator.a_output = a_output

    @staticmethod
    def set_icc_model(icc_model):
        CallGraphGenerator.icc_model = icc_model

    @staticmethod
    def exec_callgraph_flowdroid_jar():
        if CallGraphGenerator.OS == "Windows":
            process = subprocess.Popen(
                CallGraphGenerator.callgraph_flowdroid_cmd % (CallGraphGenerator.CALLGRAPH_FlowDroid_JAR_PATH,
                                                             CallGraphGenerator.apk_path,
                                                             CallGraphGenerator.semi_platform_path,
                                                             CallGraphGenerator.api_level,
                                                             CallGraphGenerator.a_output, CallGraphGenerator.icc_model),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                shell=True)
        else:
            process = subprocess.Popen(
                CallGraphGenerator.callgraph_flowdroid_cmd % (CallGraphGenerator.CALLGRAPH_FlowDroid_JAR_PATH,
                                                             CallGraphGenerator.apk_path,
                                                             CallGraphGenerator.semi_platform_path,
                                                             CallGraphGenerator.api_level,
                                                             CallGraphGenerator.a_output, CallGraphGenerator.icc_model),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                shell=True, preexec_fn=os.setsid)

        try:
            process.communicate(timeout=CallGraphGenerator.timeout)
        except Exception as e:
            process.kill()
            process.terminate()
            process.wait()

            if CallGraphGenerator.OS == "Windows":
                logging.warning("Windows CG Time Out!")
                os.system(CallGraphGenerator.win_cmd)
            else:
                logging.warning("Linux CG Time Out!")
                os.killpg(process.pid, signal.SIGTERM)
            logging.warning("FlowDroid CG Generating time out!")

    @staticmethod
    def read_dot_file(dot_path):
        CallGraphGenerator.dot_data = pydot.graph_from_dot_file(dot_path)[0]