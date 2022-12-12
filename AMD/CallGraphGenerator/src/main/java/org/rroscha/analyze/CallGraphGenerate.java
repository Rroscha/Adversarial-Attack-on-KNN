package org.rroscha.analyze;

public class CallGraphGenerate
{
    public static void main(String[] args)
    {
        if (args.length != 5)
        {
            System.out.println("Wrong number of input arguments.");
        }
        else
        {
            SootEnvironment.analyze(args[0], args[1], args[2], args[3], args[4]);

            System.out.println("CG Generated.");
        }
    }
}
