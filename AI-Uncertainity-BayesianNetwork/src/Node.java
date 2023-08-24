import java.util.*;
import java.util.regex.Pattern;

/**
 * Node class contains parent and child relationships for a node and links to the cpt table.
 * @author 220025456
 */

public class Node {
    private final Set<Node> parents = new LinkedHashSet<>();
    private final Set<Node> children = new LinkedHashSet<>();
    private final String label;
    private Factor cpt;

    /**
     * Constructor
     *
     * @param label as name of the node
     */
    public Node(String label) {
        this.label = label;
    }

    /**
     * Gets the name of the node.
     *
     * @return name of the node
     */
    public String getLabel() {
        return label;
    }

    /**
     * Adds a parent node.
     *
     * @param parent parent node
     */
    public void addParent(Node parent) {
        this.parents.add(parent);
    }

    /**
     * Adds a child node.
     *
     * @param child child node
     */
    public void addChild(Node child) {
        this.children.add(child);
    }


    /**
     * Gets the cpt table for a node
     *
     * @return cpt table
     */
    public Factor getCpt() {
        return cpt;
    }

    /**
     * Gets the parents of the node
     *
     * @return node parents
     */
    public Set<Node> getParents() {
        return parents;
    }

    /**
     * Gets the children of the node
     *
     * @return node children
     */
    public Set<Node> getChildren() {
        return children;
    }

    /**
     * Adds cpt values to the cpt table.
     *
     * @param values cpt values
     */
    public void addCPT(String values) {
        // create the factor
        cpt = new Factor(this);
        Pattern space = Pattern.compile(" ");
        String [] vals = space.split(values);
        double [] dd = new double[vals.length];
        for (int i = vals.length -1; i >= 0; i--) {
            dd[vals.length -1 -i] = Double.parseDouble(vals[i]);
        }
        cpt.addProba(dd);
    }

    /**
     * Gets all parents from the node recursively.
     *
     * @param node starting node
     * @return list of parents
     */
    public static List<Node> getAllParents(Node node) {
        List<Node> nodesList = new ArrayList<>();
        if (node.getParents().size() == 0) {
            return nodesList;
        }
        for (Node parent : node.getParents()) {
            nodesList.add(parent);
            nodesList.addAll(getAllParents(parent));
        }
        return nodesList;
    }


}
