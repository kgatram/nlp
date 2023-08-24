import java.util.Deque;
import java.util.List;

/**
 * Bidirectional Search.
 *
 * @author 220025456
 */

public class BidiSearch extends UninformedSearch {
    /**
     * push nodes to the agenda
     *
     * @param agenda search agenda
     * @param nodes    new nodes
     */
    @Override
    public void pushAgenda(Deque<Node> agenda, List<Node> nodes) {
        for (Node node : nodes) {
            agenda.addLast(node);
        }
    }

    /**
     * path cost = 1
     *
     * @param curNode current node
     * @param newCoord     new coordinates
     * @param goal         goal
     * @return cost =1 of moving to the new position
     */
    @Override
    public float pathCost(Node curNode, Coord newCoord, Coord goal) {
        return 1;
    }

}
