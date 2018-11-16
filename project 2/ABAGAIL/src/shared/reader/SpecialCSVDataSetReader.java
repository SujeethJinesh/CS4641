package shared.reader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

import shared.DataSet;
import shared.DataSetDescription;
import shared.Instance;
/**
 * Class to read in data from a CSV file without a specified label
 * @author Tim Swihart <https://github.com/chronoslynx>
 * @date 2013-03-05
 */
public class SpecialCSVDataSetReader extends SpecialDataSetReader {

    public SpecialCSVDataSetReader(String file) {
        super(file);
        // TODO Auto-generated constructor stub
    }

    @Override
    public Instance[] read() throws Exception {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        List<Instance> data = new ArrayList<>();
        Pattern pattern = Pattern.compile("[ ,]+");
        int instanceNumber = 0;
        while ((line = br.readLine()) != null) {
            String[] split = pattern.split(line.trim());
            double[] input = new double[split.length];
            for (int i = 0; i < input.length - 1; i++) {
                input[i] = Double.parseDouble(split[i]);
            }
            data.add(new Instance(input));
            data.get(instanceNumber++).setLabel(new Instance(Double.parseDouble(split[input.length - 1])));
        }
        return data.toArray(new Instance[data.size()]);
    }

}
