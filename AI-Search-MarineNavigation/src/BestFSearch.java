/**
 * Best First Search
 *
 * @author 220025456
 */

public class BestFSearch extends InformedSearch {
    /**
     * Calculates path cost
     *
     * @param current  current node
     * @param newCoord     new coordinates
     * @param goal         goal of the search problem
     * @return cost of path from new coordinates to goal
     */
    @Override
    public float pathCost(Node current, Coord newCoord, Coord goal) {
        return triangleManhattanDistance(newCoord, goal);
    }
}
