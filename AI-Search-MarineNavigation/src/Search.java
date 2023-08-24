import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * Search Abstract class.
 * This is the base class for all search algorithms implemented.
 * All search algorithms must implement `printAgenda` and `pushAgenda`.
 *
 * @author 220025456
 */
public abstract class Search {

    /**
     * print agenda information.
     *
     * @param agenda search agenda
     */
    public abstract void printAgenda(Deque<Node> agenda);

    /**
     * Add and sort nodes in agenda.
     *
     * @param agenda search agenda
     * @param nodes    new nodes
     */
    public abstract void pushAgenda(Deque<Node> agenda, List<Node> nodes);

    /**
     * Generates the successor nodes when from a list of new coordinates, and adds the valid ones to the agenda.
     *
     * @param agenda    agenda queue
     * @param extended  nodes visited
     * @param curNode   existing node to extend
     * @param newCoordinates  list of new coordinates to explore
     * @param dest        goal
     */
    public void generateNode(Deque<Node> agenda, List<Node> extended, Node curNode, ArrayList<Coord> newCoordinates, Coord dest) {
        float cost = 0;
        List<Node> allowedNodes = new ArrayList<>();
        for (Coord newCoordinate : newCoordinates) {
            boolean isVisited = checkNodes(extended, newCoordinate);
            boolean isInAgenda = checkNodes(new ArrayList<>(agenda), newCoordinate);
            if (!isVisited && !isInAgenda) {
                cost = pathCost(curNode, newCoordinate, dest);
                Node node = new Node(newCoordinate, curNode, cost );
                allowedNodes.add(node);
            }
        }
        pushAgenda(agenda, allowedNodes);
    }

    /**
     * Checks if a coordinates exist in list of nodes.
     *
     * @param nodes nodes to search
     * @param coord coordinate to find
     * @return True if exists in the nodes
     */
    public boolean checkNodes(List<Node> nodes, Coord coord) {
        return nodes.stream().map(Node::getCoord).anyMatch(s -> s.equals(coord));
    }

    /**
     * Checks if node is goal state.
     *
     * @param node input node
     * @param goal goal state
     * @return True if node is goal state
     */
    public boolean isGoal(Node node, Coord goal) {
        return node != null && node.getCoord().equals(goal);
    }

    /**
     * Calculates the cost of path.
     *
     * @param fromNode   from node
     * @param toCoord     to new coordinate
     * @param goal         goal state
     * @return cost of path
     */
    public abstract float pathCost(Node fromNode, Coord toCoord, Coord goal);

    /**
     * Gets the class name of the strategy
     *
     * @return name of the strategy
     */
    public String name() {
        return this.getClass().getSimpleName();
    }


}