import java.util.Deque;
import java.util.List;

/**
 * Breadth First Search.
 *
 * @author 220025456
 */

public class BFSearch extends UninformedSearch {
    /**
     * Add nodes to the end of agenda
     *
     * @param agenda agenda
     * @param nodes    nodes to add
     */
    @Override
    public void pushAgenda(Deque<Node> agenda, List<Node> nodes) {
        // for BFS FIFO.
        for (Node node : nodes) {
            agenda.addLast(node);
        }
    }


}
