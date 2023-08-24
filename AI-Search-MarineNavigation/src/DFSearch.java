import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;

/**
 * Depth First Search.
 * This is a search strategy that uses a LIFO agenda.
 *
 * @author 220025456
 */

public class DFSearch extends UninformedSearch {
    /**
     * push nodes to front of agenda
     *
     * @param agenda DFS agenda
     * @param nodes    nodes
     */
    @Override
    public void pushAgenda(Deque<Node> agenda, List<Node> nodes) {
        // LIFO
        for (int i = nodes.size() - 1; i >= 0; i--) {
            agenda.addFirst(nodes.get(i));
        }
    }

    /**
     * Prints all coordinates in reverse order of agenda.
     * @param agenda search agenda
     */
    @Override
    public void printAgenda(Deque<Node> agenda) {
        List<String> agendaList = new ArrayList<>(agenda.stream().map(node -> node.getCoord().toString()).toList());
        Collections.reverse(agendaList);
        System.out.println("[" + String.join(",",agendaList) + "]");
    }

}
