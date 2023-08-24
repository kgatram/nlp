import java.util.Deque;
import java.util.stream.Collectors;

/**
 * Uninformed Search.
 * This is the base class for all uninformed search algorithm - BFS, DFS and BDS.
 *
 * @author 220025456
 */
public abstract class UninformedSearch extends Search {
    /**
     * Prints all coordinates of agenda
     *
     * @param agenda search agenda
     */
    @Override
    public void printAgenda(Deque<Node> agenda) {
        String entireAgenda = agenda.stream().map(node -> node.getCoord().toString()).collect(Collectors.joining(","));
        System.out.println("[" + entireAgenda + "]");
    }

    /**
     * Calculates the cost of entering a new state. For uninformed return 1.
     *
     * @param current  current node
     * @param newCoord     new coordinates
     * @param goal         goal of the search problem
     * @return cost = 1 of moving to the new position
     */
    @Override
    public float pathCost(Node current, Coord newCoord, Coord goal) {
        return 1;
    }

}
