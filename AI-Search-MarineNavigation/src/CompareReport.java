import java.util.List;

/**
 * Generates a comparison report for a list of search results.
 *
 * @author 220025456
 */
public class CompareReport {
    private final Search algo;
    private final String config;
    private final int[][] map;
    private final Coord start;
    private final Coord goal;
    private final int steps;
    private final Node node;

    /**
     * constructor.
     *
     * @param algo search
     * @param config  configuration
     * @param map      map in configuration
     * @param start    start coordinate
     * @param goal     goal coordinate
     * @param steps    number of steps to reach goal
     * @param node     node
     */
    public CompareReport(Search algo, String config, int[][] map, Coord start, Coord goal, int steps, Node node) {
        this.algo = algo;
        this.config = config;
        this.map = map;
        this.start = start;
        this.goal = goal;
        this.steps = steps;
        this.node = node;
    }

    /**
     * manhattan distance in triangle grid.
     *
     * @param rc1 row and column of first coord
     * @param rc2 row and column of second coord
     * @return manhattan distance between coords
     */
    public float triangleManhattanDistance(Coord rc1, Coord rc2) {
        TriangleCoord abc1 = TriangleCoord.convertRowCol(rc1);
        TriangleCoord abc2 = TriangleCoord.convertRowCol(rc2);
        return Math.abs(abc1.getA() - abc2.getA()) + Math.abs(abc1.getB() - abc2.getB()) + Math.abs(abc1.getC() - abc2.getC());
    }
    /**
     * Generates a report from a list of results.
     *
     * @param rows rows to be added to the report
     * @return string format of the report along with header
     */
    public static String getRows(List<CompareReport> rows) {
        StringBuilder header = new StringBuilder("Search Algorithm,Configuration,Map Size,H-distance,Visited nodes,Cost").append("\n");
//        append each row to the header
        rows.forEach(row -> header.append(row.toString()).append("\n"));
        return header.toString();
    }

    /**
     * @return string format of search result
     */
    @Override
    public String toString() {
        return this.algo.name() + "," + config + "," + map.length + "x" + map[0].length + "," + triangleManhattanDistance(start, goal) + "," + steps + "," + node.getTotalCost() ;
    }

}
