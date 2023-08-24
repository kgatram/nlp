import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Node class contains parent info, path cost and coordinate position.
 *
 * @author 220025456
 */

public class Node implements Comparable<Node>{

    private final Node parentNode;
    private final Coord coord;
    private final float pathCost;
    public static final Node FAIL = new Node(null, null,0);

    /**
     * Constructor
     *
     * @param coord  current coordinates
     * @param parentNode parent node
     * @param pathCost   path cost
     */
    public Node(Coord coord, Node parentNode, float pathCost) {
        this.coord = coord;
        this.parentNode = parentNode;
        this.pathCost = pathCost;
    }


    /**
     * @return current state
     */

    public Coord getCoord() {
        return coord;
    }


    /**
     * @return parent node
     */
    public Node getParentNode() {
        return parentNode;
    }

    /**
     * @return cost of the node
     */
    public float getPathCost() {
        return pathCost;
    }

    /**
     * Calculates the total number of steps taken to reach the current node.
     *
     * @return number of steps taken to reach current node
     */
    public float getTotalCost() {
        return (this.parentNode != null) ? 1 + this.parentNode.getTotalCost() : 0;
    }

    /**
     * @return pathList list of coordinates traversed.
     */
    public List<Coord> getPathTraversed() {
        ArrayList<Coord> pathList = new ArrayList<>(List.of(this.coord));
        if (this.parentNode != null) {
            pathList.addAll(0, this.parentNode.getPathTraversed());
        }
        return pathList;
    }

    /**
     * Prints the path and cost of traversed nodes.
     */
    public void printPathnCost() {
        if (!this.hasNoPath()) {
            List<Coord> pathList = this.getPathTraversed();
            pathList.forEach(System.out::print);
            System.out.println();
            String directions = pathList.stream().map(Coord::getDirection).collect(Collectors.joining(" "));
            System.out.println(directions.strip());
            System.out.println(this.getTotalCost());
        } else {
            System.out.println("fail");
        }
    }

    /**
     * Checks if a path exists in this node.
     *
     * @return True if path exists
     */
    public boolean hasNoPath() {
        return coord == null && parentNode == null;
    }

    /**
     * method override to sort nodes based on path cost in ascending order
     * @param node node to compare
     * @return -1,0, 1
     */
    @Override
    public int compareTo(Node node) {
        float diff = pathCost - node.getPathCost();
        return (diff == 0) ? compareDirection(node) : (diff > 0) ? 1 : -1;
    }

    /**
     * method to compare direction of nodes in descending priority order
     * @param node node to compare
     * @return -1,0, 1
     */
    public int compareDirection(Node node) {
       int difference = this.coord.getPriority() - node.getCoord().getPriority();
         return (difference == 0) ? compareDepth(node) : (difference > 0) ? -1 : 1;
    }

    /**
     * method to compare depth of nodes in ascending order
     * depth is number of steps from start node to current node i.e. total cost in this case.
     * @param node node to compare
     * @return -1,0, 1
     */
    public int compareDepth(Node node) {
        int difference = (int) (this.getTotalCost() - node.getTotalCost());
        return (difference == 0) ? 0 : (difference > 0) ? 1 : -1;
    }
}
