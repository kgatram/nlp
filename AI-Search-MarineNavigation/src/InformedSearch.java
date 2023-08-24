import java.util.*;
import java.util.stream.Collectors;

/**
 * Informed Search.
 * This is the base class for all informed search algorithm - A* and BestF.
 *
 * @author 220025456
 */

public abstract class InformedSearch extends Search {

    /**
     * Prints entire agenda.
     *
     * @param agenda search agenda
     */
    @Override
    public void printAgenda(Deque<Node> agenda) {
        String entireAgenda = agenda.stream().map(node -> node.getCoord().toString() + ":" + node.getPathCost()).collect(Collectors.joining(","));
        System.out.println("[" + entireAgenda + "]");
    }

    /**
     * Appends nodes and orders agenda based on pathcost ascending, direction descending and Depth ascending.
     *
     * @param agenda agenda queue
     * @param nodes  nodes to add to agenda
     */
    @Override
    public void pushAgenda(Deque<Node> agenda, List<Node> nodes) {
        // add all nodes to agenda
        agenda.addAll(nodes);
        // sorted by ascending path cost
        ArrayDeque<Node> sorted = agenda.stream().sorted().collect(Collectors.toCollection(ArrayDeque::new));
        agenda.clear();
        agenda.addAll(sorted);
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


}
