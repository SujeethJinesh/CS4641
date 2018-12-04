import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Date;
import java.util.zip.GZIPInputStream;

import javax.swing.JFileChooser;
import javax.swing.JFrame;

public class DataLogger
{
    public Maze myMaze;
    double precision = 0.001;
    double pjog = 0.3;
    
    double learningRate = 0.7;
    double epsilon = 0.1;
    boolean decayingLR = true;
    
    public DataLogger()
	{
	    JFileChooser fc = new JFileChooser("./mazes/");
	    int returnVal = fc.showOpenDialog(new JFrame());
	    if (returnVal == JFileChooser.APPROVE_OPTION) {
	        try {
	            File file = fc.getSelectedFile();
	            FileInputStream fis = new FileInputStream(file);
	            GZIPInputStream gzis = new GZIPInputStream(fis);
	            ObjectInputStream in = new ObjectInputStream(gzis);
	            myMaze = (Maze)in.readObject();
	            in.close();
	        }
	        catch(Exception e) {
                Utility.show(e.getMessage());
            }
	    }
	}
    
    public void logValueIteration()
    {
        ValueIteration valItr = new ValueIteration(myMaze,pjog,precision);
        int numIterations = 0;
        while(!valItr.step()) {
            ValueFunction valuefunc = valItr.getValueFunction();
            System.out.println(++numIterations);
            System.out.print(0+"\t");
            valuefunc.displayValues();
        }
        System.out.println("Value Iteration Completed");
    }

    public void logPolicyIteration() {
        PolicyIteration polItr = new PolicyIteration(myMaze, pjog, precision, 5000, 5000);
        int numIterations = 0;
        while (!polItr.step()) {
            ValueFunction valuefunc = polItr.getValueFunction();
            System.out.println(++numIterations);
            System.out.print(0 + "\t");
            valuefunc.displayValues();
        }
//        ValueFunction valuefunc = polItr.getValueFunction();
//        System.out.println(0+"\t");
//        valuefunc.displayValues();
    }

    public void logQLearning(int series, int cycles)
    {
        
        QLearning ql = new QLearning(myMaze,pjog,learningRate,epsilon,decayingLR);
        long startTime = new Date().getTime();
        for (int i=0; i<series; i++) {
            for (int j=0; j<cycles; j++) {
                while(!ql.step())
                    ;
                long endTime = new Date().getTime();
                System.out.print((endTime-startTime)+"\t");
                System.out.print(ql.evalPolicy()+"\n");
            }
        }
    }
    
    public static void main(String[] args)
    {
        DataLogger dl = new DataLogger();

        System.out.println("Value Iteration");
        dl.logValueIteration();
//
//        System.out.println("Policy Iteration");
//        dl.logPolicyIteration();

//        System.out.println("Q Learning");
//        dl.logQLearning(5, 100);
    }
}
