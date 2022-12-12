package org.rroscha.analyze;

// Java packages
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.lang.*;

// Third party packages
import soot.*;
import soot.jimple.Constant;
import soot.jimple.*;
import soot.jimple.Expr;
import soot.jimple.infoflow.InfoflowConfiguration;
import soot.jimple.infoflow.android.InfoflowAndroidConfiguration;
import soot.jimple.infoflow.android.SetupApplication;
import soot.jimple.internal.*;
import soot.jimple.JimpleBody;
import soot.jimple.toolkits.callgraph.CallGraph;
import soot.jimple.toolkits.callgraph.CallGraphBuilder;
import soot.options.Options;
import soot.jimple.toolkits.callgraph.Edge;
import soot.util.StringTools;
import soot.util.dot.DotGraph;
import soot.util.dot.DotGraphAttribute;
import soot.util.dot.DotGraphNode;
import soot.util.queue.QueueReader;
import org.rroscha.util.AndroidUtil;



public class SootEnvironment
{
    /* ******************************************************************************************** */
    /* ************************************      Fields       ************************************* */
    /* ******************************************************************************************** */

    // Private Fields
    private static String apkPath = "";
    private static final Logger logger = Logger.getLogger("example");
    {
        logger.setLevel(Level.INFO);
    }

    // Public Fields
    public static String decompileDirectoryPath = "";
    public static String outputApkDirectory = "repackaged_apks";
    public static String apkFileName = "";
    public static String dftAPILevel = "29";   // The default API level
    public static String minAPILevel = "";
    public static String tgtAPILevel = "";
    public static String appAPILevel = "28";
    public static String packageName = "";
    public static CallGraph callGraph = null;
    public static DotGraph dot = null;

    public final static String fixAppAPILevel = appAPILevel;
    public final static String DOT_EXTENSION = ".dot";
    public final static String FLOWDROID = "_flowdroid";




    /* ******************************************************************************************** */
    /* ************************************      Methods      ************************************* */
    /* ******************************************************************************************** */

    /**
     * 'void determineApkFileName()' method sets the target apk file name.
     *
     * @return No return value.
     */
    private static void determineApkFileName()
    {
        assert !apkPath.equals("");

        if (apkPath.contains("/"))
        {
            int sliceNumber = apkPath.split("/").length;
            apkFileName = apkPath.split("/")[sliceNumber - 1];
        }
        else
        {
            int sliceNumber = apkPath.split("\\\\").length;
            apkFileName = apkPath.split("\\\\")[sliceNumber - 1];
        }

    }

    /**
     * 'void determineDecompileDirectoryPath()' method sets the directory path that stores the decompile items.
     *
     * @return No return value.
     */
    private static void determineDecompileDirectoryPath()
    {
        assert !apkPath.equals("");  // Make sure that the apkPath is not empty

        decompileDirectoryPath = apkPath.replace(".apk", "");  // Construct the output path string
    }

    /**
     * 'String determineSootClassPath(String platformPath)' method sets the android sdk path.
     * @param platformPath: Type-String. pre-path of your android sdk path.
     * @return String. Return 'SootClassPath', which is android sdk's relative or absolute path.
     */
    public static String determineSootClassPath(String platformPath)
    {
        return platformPath + "/" + "android-" + appAPILevel + "/" + "android.jar";  // File.separator
    }

    /**
     * Set up.
     * @param apkPath: Apk file path.
     * @param platformPath: pre-path of your android sdk path. E.g., '/home/user/android_sdk/android-29/android.jar''s '/home/user/android_sdk'
     * @return No return value.
     */
    public static void init(String apkPath, String platformPath, String newAppAPILevel, String iccModel)
    {
        final InfoflowAndroidConfiguration config = new InfoflowAndroidConfiguration();

        if (newAppAPILevel != null)
        {
            appAPILevel = newAppAPILevel;
        }
        else
        {
            appAPILevel = fixAppAPILevel;
        }

        SootEnvironment.apkPath = apkPath;
        determineApkFileName();
        determineDecompileDirectoryPath();

        logger.info("[DEBUG] API level -> " + appAPILevel);
        G.reset();

        String androidJar = determineSootClassPath(platformPath);

        config.getAnalysisFileConfig().setTargetAPKFile(apkPath);
        config.getAnalysisFileConfig().setAndroidPlatformDir(androidJar);

        config.setCodeEliminationMode(InfoflowConfiguration.CodeEliminationMode.NoCodeElimination);

        config.setEnableReflection(true);

        config.setCallgraphAlgorithm(InfoflowConfiguration.CallgraphAlgorithm.SPARK); // CHA or SPARK

        if (!iccModel.equals("NULL"))
        {
            config.getIccConfig().setIccModel(iccModel);
            config.getIccConfig().setIccResultsPurify(false);
        }

        SetupApplication app = new SetupApplication(config);

        app.constructCallgraph();
        CallGraph callGraph = Scene.v().getCallGraph();

        logger.info("[INFO] Edge size:\t" + callGraph.size());

        SootEnvironment.callGraph = callGraph;
    }

    /**
     * Iterate over the call Graph by visit edges one by one.
     * @param dot dot instance to create a dot file
     * @param callGraph call graph
     */
    public static void analyzeCallGraph(CallGraph callGraph, DotGraph dot, String semiDestination)
    {
        QueueReader<Edge> edges = callGraph.listener();
        Set<String> visited = new HashSet<>();

        boolean isAndroidMethod = false;
        while (edges.hasNext())
        {
            Edge edge = edges.next();
            SootMethod target = (SootMethod)edge.getTgt();
            MethodOrMethodContext source = edge.getSrc();

            // Caller
            if (!visited.contains(source.toString()) &&
                    !source.toString().equals("<dummyMainClass: void dummyMainMethod(java.lang.String[])>"))
            {
                // Store node
                dot.drawNode(source.toString());
                visited.add(source.toString());
            }

            // Callee
            if (!visited.contains(target.toString()))
            {
                // Store node
                dot.drawNode(target.toString());
                visited.add(target.toString());
            }

            // Edge
            if (!source.toString().equals("<dummyMainClass: void dummyMainMethod(java.lang.String[])>"))
            {
                dot.drawEdge(source.toString(), target.toString());
            }
        }

        String destination = semiDestination +
                File.separator +
                SootEnvironment.apkFileName.substring(0, SootEnvironment.apkFileName.lastIndexOf("."));

        logger.info("plot dot file.");
        dot.plot(destination + FLOWDROID +DOT_EXTENSION);
    }

    public static void analyze(String apkPath, String semiPlatformPath, String apiLevel, String semiDestination,
                               String iccModel)
    {
        SootEnvironment.init(apkPath, semiPlatformPath, apiLevel, iccModel);

        // Generate call graph
        dot = new DotGraph(SootEnvironment.apkFileName);
        analyzeCallGraph(SootEnvironment.callGraph, dot, semiDestination);
    }
}


