/**
 * A* search algorithm
 *
 * @author 220025456
 */

public class AStarSearch extends InformedSearch{
    /**
     * Calculates the path cost = cost from start to new coordinate + heuristic from new coordinate to goal
     *
     * @param curNode  current node
     * @param newCoord successor coordinate
     * @param goal     goal coordinate
     * @return cost of moving to the new state
     */
    @Override
    public float pathCost(Node curNode, Coord newCoord, Coord goal) {
        return (curNode == null ? 0 : curNode.getTotalCost() + 1)
                + triangleManhattanDistance(newCoord, goal);
    }

}
