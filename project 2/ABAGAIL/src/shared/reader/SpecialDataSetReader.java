package shared.reader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

import shared.DataSet;
import shared.Instance;

/**
 * An instance reader reads instances from a file
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public abstract class SpecialDataSetReader {
    /**
     * The files to read from
     */
    protected String file;

    /**
     * Make a new instance reader
     * @param file the file to read from
     */
    public SpecialDataSetReader(String file) {
        this.file = file;
    }

    /**
     * Read the thing
     * @return the data
     */
    public abstract Instance[] read() throws Exception;

}
