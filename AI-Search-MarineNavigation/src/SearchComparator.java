import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Search Comparator class.
 * This class is used to compare all the search algorithm implemented.
 *
 * @author 220025456
 */

public class SearchComparator {
    private final List<? extends Search> searchList = List.of(new BFSearch(), new DFSearch(), new BestFSearch(), new AStarSearch(), new BidiSearch());

    /**
     * Compare search algorithm for given configurations.
     */
    public void compareConfig() {
        List<CompareReport> report = new ArrayList<>();
        for (Conf conf : Conf.values()) {
            searchList.forEach(algo -> {
                report.add(getResults(algo, conf.name(), conf.getMap().getMap(), conf.getS(), conf.getG()));
            });
        }
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter("report.csv", false));
            writer.append(CompareReport.getRows(report));
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    /**
     * get the results for a given search algorithm
     *
     * @param algo search to use
     * @param config name of configuration
     * @param start    start coordinate
     * @param goal     goal coordinate
     * @param map      map in configuration
     * @return search result
     */
    public CompareReport getResults(Search algo, String config, int[][] map, Coord start, Coord goal ) {
        Vessel ferry = new Vessel(start, goal, map, algo);
        ferry.navigate();
        return new CompareReport(algo, config, map, start, goal, ferry.getSteps(), ferry.getSuggestedNode());
    }

}
