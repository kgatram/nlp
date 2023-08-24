import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * This class severs as an agent that will navigate the map using any of the implemented search algorithms.
 *
 * @author 220025456
 */

public class Vessel {
    private static final int LAND = 1;
    protected int steps;
    protected Node suggestedNode;
    protected final Coord start;
    protected final Coord goal;
    protected final int[][] map;
    protected final Search algo;
    protected ArrayDeque<Node> agenda = new ArrayDeque<>();
    protected ArrayList<Node> expandedNodes = new ArrayList<>();
    protected boolean diagonalMoves;


    /**
     * Constructor.
     *
     * @param start departure or start point
     * @param goal  destination or end point
     * @param map   map to navigate
     * @param algo  search algorithm to use.
     */
    public Vessel(Coord start, Coord goal, int[][] map, Search algo) {
        this.start = start;
        this.goal = goal;
        this.map = map;
        this.algo = algo;
        this.steps = 0;
        this.suggestedNode = Node.FAIL;
        this.diagonalMoves = false;
    }


    /**
     * Navigation function.
     */
    public void navigate() {
        if (algo instanceof BidiSearch) {
            bidirectional();
        } else {
            unidirectional();
        }
    }

    /**
     * search for uni directional algorithms
     */
    public void unidirectional() {
        float cost = algo.pathCost(null, start, goal);
        agenda.add(new Node(start, null, cost));
        int step = 0;
        Node resultNode = Node.FAIL;

        while (!agenda.isEmpty()) {
            step++;
            Node node = sail();
            if (node != null) {
                resultNode = node;
                break;
            }
        }
        setSuggestedNode(resultNode);
        setSteps(step);
        // show results
        showPathnCost(resultNode, step);
    }

    /**
     * Search simultaneously from both the start and goal nodes.
     */
    public void bidirectional() {
        // ferry navigating from start to goal
        Vessel ferryForward = new Vessel(start, goal, map, algo);
        ferryForward.setDiagonalMoves(diagonalMoves);

        // ferry navigating from goal to start
        Vessel ferryBackward = new Vessel(goal, start, map, algo);
        ferryBackward.setDiagonalMoves(diagonalMoves);

        // set initial nodes to the agenda
        ferryForward.getAgenda().add(new Node(start, null, 0));
        ferryBackward.getAgenda().add(new Node(goal, null, 0));

        int steps = 0;
        Node resultNode = Node.FAIL;
        while (!ferryForward.getAgenda().isEmpty() && !ferryBackward.getAgenda().isEmpty()) {
            steps++;
            System.out.print("[S->] ");
            resultNode = ferryForward.sail();
            if (resultNode != null) {
                break;
            }
            steps++;
            System.out.print("[<-G] ");
            resultNode = ferryBackward.sail();
            if (resultNode != null) {
                break;
            } else {
                // check if both ferries visited the same node
               resultNode = checkCommonNode(ferryForward.getExpandedNodes(), ferryBackward.getExpandedNodes());
                if (resultNode != null) {
                    break;
                }
            }
        }
        // if no path found
        resultNode =  (resultNode == null) ? Node.FAIL : resultNode;

        // ... for evaluation purposes
        setSuggestedNode(resultNode);
        setSteps(steps);
        // show results
        showPathnCost(resultNode, steps);
    }

    /**
     * sail thru agenda.
     *
     * @return suggested nodes
     */
    public Node sail() {
        algo.printAgenda(agenda);
        Node curNode = agenda.pollFirst();
        if (algo.isGoal(curNode, goal)) {
            return curNode;
        } else {
            expandedNodes.add(curNode);
            ArrayList<Coord> validCoord = checkAllowedMoves(curNode.getCoord());
            algo.generateNode(agenda, expandedNodes, curNode, validCoord, goal);
        }
        return null;
    }

    /**
     * Check if state is out of bounds.
     *
     * @param coord coordinate in the map
     * @return True if state is v
     */
    public boolean isOutofMap(Coord coord) {
        int rowLimits = map.length;
        int colLimits = map[0].length;
        int row = coord.getR();
        int col = coord.getC();
        //  check if state does not exist in the map( i.e. out of bounds), and if state is on land
        return row < 0 || row >= rowLimits || col < 0 || col >= colLimits;
    }

    /**
     * Checks if a coord is land.
     *
     * @param coord coordinate in the map
     * @return True if state is on land
     */
    public boolean isLand(Coord coord) {
        return this.map[coord.getR()][coord.getC()] == LAND;
    }

    /**
     * Gets the valid coord that an agent can move to from a current position.
     *
     * @param coord current coordinate
     * @return valid successor coordinates
     */
    private ArrayList<Coord> checkAllowedMoves(Coord coord) {
        ArrayList<Coord> validNewCoord = new ArrayList<>();
        Moves.getAllowedMoves(coord, diagonalMoves).forEach(mv -> {
            Coord newCoord = mv.move(coord);
            if (!isOutofMap(newCoord) && !isLand(newCoord)) {
                validNewCoord.add(newCoord);
            }
        });
        return validNewCoord;
    }

    /**
     * check any common node explored by both the ferries.
     *
     * @param exploredForward explored node in forward direction
     * @param exploredBackward explored node in backward direction
     * @return common node
     */
    public Node checkCommonNode(List<Node> exploredForward, List<Node> exploredBackward) {
        List<Coord> fCoords = exploredForward.stream().map(Node::getCoord).collect(Collectors.toList());
        List<Coord> bCoords = exploredBackward.stream().map(Node::getCoord).collect(Collectors.toList());
        fCoords.retainAll(bCoords);

        if (fCoords.size() > 0) {
            Coord common = fCoords.get(0);
            System.out.println("Connecting at " + common);
            Optional<Node> fCommonNode = exploredForward.stream().filter(node -> node.getCoord().equals(common)).findFirst();
            Optional<Node> bCommonNode = exploredBackward.stream().filter(node -> node.getCoord().equals(common)).findFirst();
            if (fCommonNode.isPresent() && bCommonNode.isPresent()) {
                return connectNodes(fCommonNode.get(), bCommonNode.get());
            }
        }
        return null;
    }

    /**
     * Connect at common node.
     *
     * @param forwardNode parent node
     * @param backwardNode common node
     * @return merged node
     */
    private Node connectNodes(Node forwardNode, Node backwardNode) {
        Node connectedNode = forwardNode.getParentNode();
        Node node = backwardNode;
        node.getCoord().setDirection(forwardNode.getCoord().getDirection() + " " + node.getCoord().getDirection());
        while (node != null) {
            Coord newCoord = node.getCoord();
            connectedNode = new Node(newCoord, connectedNode, algo.pathCost(connectedNode, newCoord, goal));
            node = node.getParentNode();
        }
        return connectedNode;
    }

    /**
     * @return steps
     */
    public int getSteps() {
        return steps;
    }

    /**
     * Sets the steps
     *
     * @param steps the steps to set
     */
    public void setSteps(int steps) {
        this.steps = steps;
    }

    /**
     * @return suggested node
     */
    public Node getSuggestedNode() {
        return suggestedNode;
    }

    /**
     * Sets the  node
     *
     * @param node the suggested node to set
     */
    public void setSuggestedNode(Node node) {
        this.suggestedNode = node;
    }

    /**
     * Option to use/ not use advanced features implemented
     *
     * @param diagonalMoves option to toggle use of advanced features.
     */
    public void setDiagonalMoves(boolean diagonalMoves) {
        this.diagonalMoves = diagonalMoves;
    }

    /**
     * @return agenda
     */
    public ArrayDeque<Node> getAgenda() {
        return agenda;
    }


    /**
     * @return expanded nodes
     */
    public ArrayList<Node> getExpandedNodes() {
        return expandedNodes;
    }

    /**
     * Log the path and the steps taken to get to a node.
     *
     * @param node  node to get path from
     * @param steps number of steps taken to get to the node
     */
    public void showPathnCost(Node node, int steps) {
        node.printPathnCost();
        System.out.println(steps);
    }



}
