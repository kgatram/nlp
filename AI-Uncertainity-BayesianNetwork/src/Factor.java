import java.util.*;

/**
 * Factor class containing the probability table and the variables or labels used in the probability table.
 * @author 220025456
 */
public class Factor {
    private final Map<String, Double> cpt = new TreeMap<>();
    private final Set<Node> nodeSet = new LinkedHashSet<>();

    /**
     * Constructor with single node.
     * @param node node
     */
    public Factor(Node node) {
        nodeSet.addAll(node.getParents());
        nodeSet.add(node);
    }

    /**
     * Constructor with multiple nodes.
     *
     * @param nodes set of nodes
     */
    public Factor(Set<Node> nodes) {
        this.nodeSet.addAll(nodes);
    }

    /**
     * Gets the cpt.
     *
     * @return cpt as a map
     */
    public Map<String, Double> getCpt() {
        return cpt;
    }

    /**
     * Gets all the nodes involved in cpt.
     *
     * @return set of nodes
     */
    public Set<Node> getNodeSet() {
        return nodeSet;
    }

    /**
     * Add probability to CPT with key as "00", "01", "10", "11", ...
     *
     * @param values probabilities
     */
    public void addProba(double... values) {
        int noOfNodes = getNodeSet().size();
        int size = (int) Math.pow(2, noOfNodes);
        if (values.length == size) {
            for (int i = 0; i < size; i++) {
                String key = binaryExpansion(Integer.toBinaryString(i), noOfNodes);
                cpt.put(key, values[i]);
            }
        }
    }

    /**
     * Generates a list of truth table for number of nodes involved.
     * used to generate the probability table.
     * 00, 01, 10, 11, ...
     *
     * @return truth table
     */
    public List<boolean[]> truthTable() {
        int noOfNodes = getNodeSet().size();
        int size = (int) Math.pow(2, noOfNodes); // 2^noOfNodes size of the truth table
        List<boolean[]> table = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            String key = binaryExpansion(Integer.toBinaryString(i), noOfNodes);
            char[] keyCharacters = key.toCharArray();
            boolean[] row = new boolean[noOfNodes];
            for (int j = 0; j < keyCharacters.length; j++) {
                row[j] = keyCharacters[j] == '1';
            }
            table.add(row);
        }
        return table;
    }

    /**
     * returns 0,1,2,3 as 00,01,10,11
     *
     * @param binaryString Integers as string
     * @param size         expansion size
     * @return expanded binary string of size
     */
    private String binaryExpansion(String binaryString, int size) {
        if (size > binaryString.length()) {
            return "0".repeat(size - binaryString.length()) + binaryString;
        }
        return binaryString;
    }


    /**
     * coverts a string key 00, 01, 10, 11, ... to boolean True, False
     *
     * @param key cpt key
     * @return boolean array
     */
    public boolean[] keyToBoolean(String key) {
        boolean[] booleanKey = new boolean[key.length()];
        char[] keyArray = key.toCharArray();
        for (int i = 0; i < keyArray.length; i++) {
            booleanKey[i] = keyArray[i] == '1';
        }
        return booleanKey;
    }

    /**
     * Gets key value as 00, 01, 10, 11, ... for size of the node set.
     *
     * @param nodeValues values as true false
     * @return cpt key as 00, 01, 10, 11, ...
     */
    public String getKey(boolean[] nodeValues) {
        String key = "";
        if (nodeValues.length == getNodeSet().size()) {
            for (int i = 0; i < nodeValues.length; i++) {
                key += (nodeValues[i]) ? "1" : "0";
            }
        }
        return key;
    }

    /**
     * get cpt value by key 00, 01, 10, 11, ...
     *
     * @param labelTrueFalse label A and value true, label B and value false
     * @return cpt for the given key
     */
    public double getCPTbyKey(Map<String, Boolean> labelTrueFalse) {
        String key = "";
        for (Node node : getNodeSet()) {
            key += (labelTrueFalse.get(node.getLabel())) ? "1" : "0";
        }
        return cpt.get(key);
    }

    /**
     * Generates a copy of a factor.
     *
     * @return copy of a factor
     */
    public Factor copy() {
        Factor factor = new Factor(getNodeSet());
        getCpt().forEach((key, prob) -> factor.assign(keyToBoolean(key), prob));
        return factor;

    }

    /**
     * Does factor includes input node?
     *
     * @param node2 node to check
     * @return True if exist
     */
    public boolean includes(Node node2) {
        return getNodeSet().contains(node2);
    }

    /**
     * A map of node and its boolean value.
     *
     * @param values boolean values
     * @return boolean value mapped to node
     */
    public Map<String, Boolean> getKeyValue(boolean[] values) {
        Map<String, Boolean> labelKeyValue = new HashMap<>();
        Node[] nodes = getNodeSet().toArray(Node[]::new);
        for (int i = 0; i < values.length; i++) {
            labelKeyValue.put(nodes[i].getLabel(), values[i]);
        }
        return labelKeyValue;
    }

    /**
     * Joins two factors
     *
     * @param f2 second factor
     * @return joined factor f3 = f1 * f2
     */
    public Factor join(Factor f2) {
        Set<Node> f1nodes = this.getNodeSet();
        Set<Node> f2nodes = f2.getNodeSet();

        // new factor f3 is the union of f1 and f2 nodes
        Set<Node> f12 = new LinkedHashSet<>(f1nodes);
        f12.addAll(f2nodes);
        Factor f3 = new Factor(f12);
        // generate truth table for new factor f3
        for (boolean[] truthrow : f3.truthTable()) {
            Map<String, Boolean> labelKeyValueMap = f3.getKeyValue(truthrow);
            // f3 = f1 * f2
            f3.assign(truthrow, this.getCPTbyKey(labelKeyValueMap) * f2.getCPTbyKey(labelKeyValueMap));
        }
        return f3;
    }

    /**
     * eliminate input node from the factor
     *
     * @param eliminateLabel input node
     * @return factor new factor with the input node eliminated
     */
    public Factor sumOut(Node eliminateLabel) {

        Set<Node> newNodes = new LinkedHashSet<>(this.getNodeSet());
        newNodes.remove(eliminateLabel);
        Factor newFactor = new Factor(newNodes);

        List<Boolean> TrueFalse = List.of(true, false);
        for (boolean[] truthRow : newFactor.truthTable()) {
            double sum = 0;
            for (Boolean tf : TrueFalse) {
                Map<String, Boolean> labelTrueFalse = newFactor.getKeyValue(truthRow);
                // add left out node
                labelTrueFalse.put(eliminateLabel.getLabel(), tf);
                // get probability and add to the sum
                sum += getCPTbyKey(labelTrueFalse);
            }
            newFactor.assign(truthRow, sum);
        }
        return newFactor;
    }

    /**
     * Zeroies the probability of a node. Use for evidence not in the query.
     *
     * @param node  node to zeroise
     * @param value boolean value to zeroise
     */
    public void zeroised(Node node, boolean value) {
        for (boolean[] truthRow : truthTable()) {
            Map<String, Boolean> labelTF = getKeyValue(truthRow);
            if (labelTF.get(node.getLabel()) == value) {
                assign(truthRow, 0.0);
            }
        }
    }

    /**
     * Normalizes the values in the probability table.
     */
    public void normalize() {
        if (nodeSet.size() == 1) {
            double total = cpt.values().stream().reduce(0.0, Double::sum);
            cpt.forEach((key, proba) -> assign(keyToBoolean(key), proba / total));
        }
    }

    /**
     * assign a probability.
     *
     * @param values truth row
     * @param proba   probability value
     */
    public void assign(boolean[] values, double proba) {
        if (values.length == getNodeSet().size()) {
            cpt.put(getKey(values), proba);
        }
    }

}
